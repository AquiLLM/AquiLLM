from __future__ import annotations

import os
import socket
import uuid
from types import SimpleNamespace

import pytest
from django.conf import settings
from django.db import IntegrityError
from django.db.models.deletion import Collector


def _database_is_reachable() -> bool:
    database = settings.DATABASES["default"]
    try:
        with socket.create_connection(
            (database["HOST"], int(database.get("PORT") or 5432)), timeout=0.2
        ):
            return True
    except OSError:
        return False


database_required = pytest.mark.skipif(
    not _database_is_reachable() and os.environ.get("KG_REQUIRE_POSTGRES_TESTS") != "1",
    reason="configured PostgreSQL database is not reachable",
)


@pytest.mark.parametrize(
    "parent_identity",
    (
        (None,) * 10,
        (
            17,
            29,
            uuid.uuid4(),
            None,
            None,
            None,
            None,
            29,
            None,
            None,
        ),
    ),
)
def test_figure_delete_signal_fences_the_exact_collected_parent_triple(
    monkeypatch,
    parent_identity,
):
    from apps.knowledge_graph.graph import invalidation

    calls = []
    (
        parent_content_type_id,
        parent_object_pkid,
        parent_object_id,
        parent_handwritten_notes_document_id,
        parent_image_upload_document_id,
        parent_media_upload_document_id,
        parent_pdf_document_id,
        parent_raw_text_document_id,
        parent_tex_document_id,
        parent_vtt_document_id,
    ) = parent_identity
    instance = SimpleNamespace(
        id=uuid.uuid4(),
        pkid=41,
        collection_id=7,
        full_text_hash="a" * 64,
        parent_content_type_id=parent_content_type_id,
        parent_object_pkid=parent_object_pkid,
        parent_object_id=parent_object_id,
        parent_handwritten_notes_document_id=parent_handwritten_notes_document_id,
        parent_image_upload_document_id=parent_image_upload_document_id,
        parent_media_upload_document_id=parent_media_upload_document_id,
        parent_pdf_document_id=parent_pdf_document_id,
        parent_raw_text_document_id=parent_raw_text_document_id,
        parent_tex_document_id=parent_tex_document_id,
        parent_vtt_document_id=parent_vtt_document_id,
    )
    sender = SimpleNamespace(
        _meta=SimpleNamespace(label_lower="apps_documents.documentfigure")
    )
    monkeypatch.setattr(
        invalidation,
        "cleanup_document_graph_state",
        lambda *args, **kwargs: calls.append((args, kwargs))
        or invalidation.DocumentGraphCleanupResult(
            affected_collection_ids=(7,),
            current_collection_id=7,
            source_hash="a" * 64,
            ingestion_complete=True,
            has_active_document_artifact=False,
        ),
    )
    monkeypatch.setattr(
        invalidation,
        "_ensure_origin_delete_scope_locked",
        lambda origin, *, using: SimpleNamespace(
            snapshot=SimpleNamespace(locked_collection_ids=(7,))
        ),
    )
    monkeypatch.setattr(
        invalidation,
        "_assert_document_delete_scope",
        lambda *_args: None,
    )

    invalidation.document_pre_delete(
        sender,
        instance,
        using="default",
        origin=instance,
    )

    assert calls[0][1]["expected_parent_identity"] == parent_identity


def test_collection_delete_signal_fences_the_collected_parent_id(monkeypatch):
    from apps.knowledge_graph.graph import invalidation
    from apps.knowledge_graph.projection import lifecycle
    calls, fences = [], []
    instance = SimpleNamespace(pk=23, parent_id=19)
    context = SimpleNamespace(snapshot=SimpleNamespace(locked_collection_ids=(23,)))
    monkeypatch.setattr(
        invalidation,
        "cleanup_collection_graph_state",
        lambda *args, **kwargs: calls.append((args, kwargs)) or (23,),
    )
    monkeypatch.setattr(
        invalidation,
        "_ensure_origin_delete_scope_locked",
        lambda origin, *, using: context,
    )
    monkeypatch.setattr(
        invalidation, "_assert_collection_delete_scope", lambda *_: None
    )
    monkeypatch.setattr(
        lifecycle,
        "tombstone_collection_projections_locked",
        lambda *, collection_id, now, using: fences.append((collection_id, using)),
    )
    invalidation.collection_pre_delete(
        object(), instance, using="default", origin=instance
    )

    assert calls == [
        (
            ((23,),),
            {
                "reason": "collection_deleted",
                "using": "default",
                "all_artifacts": True,
                "expected_parent_id": 19,
                "_origin_context": context,
            },
        )
    ]
    assert fences == [(23, "default")]


@pytest.mark.django_db(transaction=True)
@database_required
def test_stale_parent_collector_cannot_delete_a_reparented_figure():
    from django.contrib.auth.models import User

    from apps.collections.models import Collection
    from apps.documents.models import DocumentFigure, RawTextDocument

    user = User.objects.create_user(
        username=f"kg-figure-fence-{uuid.uuid4()}", password="unused"
    )
    collection = Collection.objects.create(name=f"figure fence {uuid.uuid4()}")
    parent_a = RawTextDocument(
        title="parent a",
        full_text="parent a",
        collection=collection,
        ingested_by=user,
    )
    parent_a.save(dont_rechunk=True)
    parent_b = RawTextDocument(
        title="parent b",
        full_text="parent b",
        collection=collection,
        ingested_by=user,
    )
    parent_b.save(dont_rechunk=True)
    figure = DocumentFigure(
        title="figure",
        full_text="caption",
        collection=collection,
        ingested_by=user,
        source_format="pdf",
    )
    figure.parent_document = parent_a
    figure.save(dont_rechunk=True)

    collector = Collector(using="default", origin=parent_a)
    collector.collect((parent_a,))
    current_figure = DocumentFigure.objects.get(pk=figure.pk)
    current_figure.parent_document = parent_b
    current_figure.save(
        dont_rechunk=True,
        update_fields=["parent_object_id"],
    )

    with pytest.raises(RuntimeError, match="changed during lifecycle deletion"):
        collector.delete()

    assert RawTextDocument.objects.filter(pk=parent_a.pk).exists()
    figure.refresh_from_db()
    assert figure.parent_document == parent_b

    RawTextDocument.objects.get(pk=parent_a.pk).delete()
    assert DocumentFigure.objects.filter(pk=figure.pk).exists()


@pytest.mark.django_db(transaction=True)
@database_required
def test_stale_parent_collector_cannot_delete_a_reparented_collection():
    from django.contrib.auth.models import User

    from apps.collections.models import Collection
    from apps.documents.models import RawTextDocument

    user = User.objects.create_user(
        username=f"kg-collection-fence-{uuid.uuid4()}", password="unused"
    )
    parent_a = Collection.objects.create(name=f"collection a {uuid.uuid4()}")
    parent_b = Collection.objects.create(name=f"collection b {uuid.uuid4()}")
    child = Collection.objects.create(
        name=f"collection child {uuid.uuid4()}", parent=parent_a
    )
    document = RawTextDocument(
        title="child document",
        full_text="must survive stale collection deletion",
        collection=child,
        ingested_by=user,
    )
    document.save(dont_rechunk=True)

    collector = Collector(using="default", origin=parent_a)
    collector.collect((parent_a,))
    current_child = Collection.objects.get(pk=child.pk)
    current_child.parent = parent_b
    current_child.save(update_fields=["parent"])

    with pytest.raises(RuntimeError, match="changed during lifecycle deletion"):
        collector.delete()

    assert Collection.objects.filter(pk=parent_a.pk).exists()
    child.refresh_from_db()
    assert child.parent_id == parent_b.pk

    Collection.objects.get(pk=parent_a.pk).delete()
    assert Collection.objects.filter(pk=child.pk).exists()
    assert RawTextDocument.objects.filter(pk=document.pk).exists()


@pytest.mark.django_db(transaction=True)
@database_required
def test_stale_parent_collector_cannot_orphan_a_late_attached_figure():
    from django.contrib.auth.models import User

    from apps.collections.models import Collection
    from apps.documents.models import DocumentFigure, RawTextDocument

    user = User.objects.create_user(
        username=f"kg-figure-join-fence-{uuid.uuid4()}", password="unused"
    )
    collection = Collection.objects.create(name=f"figure join fence {uuid.uuid4()}")
    parent = RawTextDocument(
        title="parent",
        full_text="parent",
        collection=collection,
        ingested_by=user,
    )
    parent.save(dont_rechunk=True)
    collector = Collector(using="default", origin=parent)
    collector.collect((parent,))

    figure = DocumentFigure(
        title="late figure",
        full_text="late caption",
        collection=collection,
        ingested_by=user,
        source_format="pdf",
    )
    figure.parent_document = parent
    figure.save(dont_rechunk=True)

    with pytest.raises((RuntimeError, IntegrityError)):
        collector.delete()

    assert RawTextDocument.objects.filter(pk=parent.pk).exists()
    assert DocumentFigure.objects.filter(pk=figure.pk).exists()

    RawTextDocument.objects.get(pk=parent.pk).delete()
    assert not DocumentFigure.objects.filter(pk=figure.pk).exists()
