from __future__ import annotations

import os
import socket
import uuid
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
from django.conf import settings
from django.db import OperationalError, close_old_connections, transaction


def _database_is_reachable() -> bool:
    database = settings.DATABASES["default"]
    try:
        with socket.create_connection(
            (database["HOST"], int(database.get("PORT") or 5432)),
            timeout=0.2,
        ):
            return True
    except OSError:
        return False


database_required = pytest.mark.skipif(
    not _database_is_reachable() and os.environ.get("KG_REQUIRE_POSTGRES_TESTS") != "1",
    reason="configured PostgreSQL database is not reachable",
)


def _user():
    from django.contrib.auth.models import User

    return User.objects.create_user(
        username=f"kg-lifecycle-lock-{uuid.uuid4()}",
        password="unused",
    )


def _document(*, collection, user, title):
    from apps.documents.models import RawTextDocument

    row = RawTextDocument(
        title=title,
        full_text=title,
        collection=collection,
        ingested_by=user,
    )
    row.save(dont_rechunk=True)
    return row


def _figure(*, collection, user, title, parent):
    from apps.documents.models import DocumentFigure

    row = DocumentFigure(
        title=title,
        full_text=title,
        collection=collection,
        ingested_by=user,
        source_format="pdf",
    )
    row.parent_document = parent
    row.save(dont_rechunk=True)
    return row


@pytest.mark.django_db(transaction=True)
@database_required
def test_figure_origin_snapshot_expands_to_exact_owner_and_both_collections():
    from apps.collections.models import Collection
    from apps.knowledge_graph.graph.invalidation import (
        _snapshot_origin_delete_scope,
    )

    user = _user()
    source_collection = Collection.objects.create(name=f"source {uuid.uuid4()}")
    figure_collection = Collection.objects.create(
        name=f"figures {uuid.uuid4()}",
        parent=source_collection,
    )
    parent = _document(
        collection=source_collection,
        user=user,
        title="source parent",
    )
    figure = _figure(
        collection=figure_collection,
        user=user,
        title="derived figure",
        parent=parent,
    )

    snapshot = _snapshot_origin_delete_scope(figure, using="default")

    assert snapshot.locked_collection_ids == tuple(
        sorted((source_collection.pk, figure_collection.pk))
    )
    assert tuple(row.document.document_id for row in snapshot.documents) == (figure.id,)
    assert {row.document.document_id for row in snapshot.fence_documents} == {
        parent.id,
        figure.id,
    }


@pytest.mark.django_db(transaction=True)
@database_required
def test_cross_collection_figure_parent_deletes_share_one_sorted_lock_union():
    from apps.collections.models import Collection
    from apps.documents.models import DocumentFigure, RawTextDocument

    user = _user()
    collection_a = Collection.objects.create(name=f"lock a {uuid.uuid4()}")
    collection_b = Collection.objects.create(name=f"lock b {uuid.uuid4()}")
    parent_a = _document(collection=collection_a, user=user, title="parent a")
    parent_b = _document(collection=collection_b, user=user, title="parent b")
    figure_a = _figure(
        collection=collection_b,
        user=user,
        title="figure owned by a",
        parent=parent_a,
    )
    figure_b = _figure(
        collection=collection_a,
        user=user,
        title="figure owned by b",
        parent=parent_b,
    )
    barrier = Barrier(2)

    def delete_parent(pkid):
        close_old_connections()
        try:
            with transaction.atomic():
                parent = RawTextDocument.objects.get(pkid=pkid)
                barrier.wait(timeout=10)
                parent.delete()
        finally:
            close_old_connections()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = (
            executor.submit(delete_parent, parent_a.pkid),
            executor.submit(delete_parent, parent_b.pkid),
        )
        for future in futures:
            future.result(timeout=20)

    assert not RawTextDocument.objects.filter(
        pkid__in=(parent_a.pkid, parent_b.pkid)
    ).exists()
    assert not DocumentFigure.objects.filter(
        pkid__in=(figure_a.pkid, figure_b.pkid)
    ).exists()


@pytest.mark.django_db(transaction=True)
@database_required
def test_collection_parent_move_and_delete_do_not_deadlock():
    from apps.collections.models import Collection

    parent = Collection.objects.create(name=f"move parent {uuid.uuid4()}")
    destination = Collection.objects.create(name=f"move destination {uuid.uuid4()}")
    child = Collection.objects.create(name=f"move child {uuid.uuid4()}", parent=parent)
    barrier = Barrier(2)

    def move_child():
        close_old_connections()
        try:
            with transaction.atomic():
                current = Collection.objects.get(pk=child.pk)
                barrier.wait(timeout=10)
                current.parent = destination
                current.save(update_fields=["parent"])
            return None
        except Exception as exc:  # stale-Collector failure is an allowed winner
            return exc
        finally:
            close_old_connections()

    def delete_parent():
        close_old_connections()
        try:
            with transaction.atomic():
                current = Collection.objects.get(pk=parent.pk)
                barrier.wait(timeout=10)
                current.delete()
            return None
        except Exception as exc:  # stale-Collector failure is an allowed winner
            return exc
        finally:
            close_old_connections()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = (executor.submit(move_child), executor.submit(delete_parent))
        outcomes = tuple(future.result(timeout=20) for future in futures)

    for outcome in outcomes:
        assert not (
            isinstance(outcome, OperationalError)
            and "deadlock" in str(outcome).casefold()
        )
    surviving_child = Collection.objects.filter(pk=child.pk).first()
    if surviving_child is not None:
        assert surviving_child.parent_id == destination.pk
