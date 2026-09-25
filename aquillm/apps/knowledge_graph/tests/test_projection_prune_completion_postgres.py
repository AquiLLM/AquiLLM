"""Real PostgreSQL retention, completion fencing and role-ACL regressions."""

from __future__ import annotations

import os
from types import SimpleNamespace
from uuid import uuid4

import pytest
from django.db import ProgrammingError, connection, transaction
from django.utils import timezone

from apps.collections.models import Collection
from apps.knowledge_graph.models import CollectionGraphProjection
from apps.knowledge_graph.projection import generation_audit, reconciler
from apps.knowledge_graph.projection.state_repository import (
    FunctionProjectionStateRepository,
)
from apps.knowledge_graph.tests.test_projection_locking_postgres import (
    _active_artifact,
)

pytestmark = [
    pytest.mark.django_db(transaction=True, databases="__all__"),
    pytest.mark.skipif(
        os.environ.get("KG_REQUIRE_POSTGRES_TESTS") != "1",
        reason="set KG_REQUIRE_POSTGRES_TESTS=1 for PostgreSQL integration tests",
    ),
]


@pytest.fixture
def authority(monkeypatch):
    assert connection.vendor == "postgresql"
    monkeypatch.setenv("KG_PROJECTION_IDENTIFIER_HMAC_KEY", "audit-test-key-only")
    monkeypatch.setenv("KG_PROJECTION_IDENTIFIER_KEY_VERSION", "key-v1")
    collection = Collection.objects.create(name=f"prune audit {uuid4()}")
    artifact = _active_artifact(collection)

    def create(state="superseded"):
        return CollectionGraphProjection.objects.create(
            collection=collection,
            collection_pk_snapshot=collection.pk,
            artifact=artifact,
            artifact_pk_snapshot=artifact.pk,
            state=state,
            schema_version="collection-graph-v1",
            projection_version="projection-v1",
            identifier_key_version="key-v1",
            membership_epoch=1,
            membership_checksum="a" * 64,
            private_mapping_checksum="b" * 64,
            failure_code="write_failed" if state == "failed" else "",
            superseded_at=timezone.now() if state == "superseded" else None,
        )

    return collection, create


def test_more_than_two_prune_pages_retain_the_same_newest_two(authority, monkeypatch):
    collection, create = authority
    rows = [create() for _ in range(9)]
    CollectionGraphProjection.objects.filter(collection=collection).update(
        created_at=timezone.now()
    )
    retained = set(sorted((row.id for row in rows), reverse=True)[:2])
    deleted = []
    monkeypatch.setattr(
        reconciler,
        "_memgraph_repository",
        lambda: SimpleNamespace(
            delete_generation=lambda **kwargs: (
                deleted.append(kwargs["generation_key"].value) or True
            )
        ),
    )

    counts = [
        reconciler.prune_graph_projection_generations(
            collection_id=collection.pk, page_size=2, retain=2, dry_run=False
        ).deleted_count
        for _ in range(5)
    ]

    assert counts == [2, 2, 2, 1, 0]
    assert len(set(deleted)) == 7
    assert CollectionGraphProjection.objects.filter(collection=collection).count() == 9
    assert (
        set(
            CollectionGraphProjection.objects.filter(
                collection=collection, pruned_at__isnull=True
            ).values_list("id", flat=True)
        )
        == retained
    )


def test_graph_deletion_failure_does_not_acknowledge_completion(authority, monkeypatch):
    _collection, create = authority
    row = create("failed")

    def unavailable(**kwargs):
        raise ConnectionError("simulated graph outage")

    graph = SimpleNamespace(delete_generation=unavailable)
    monkeypatch.setattr(reconciler, "_memgraph_repository", lambda: graph)
    with pytest.raises(ConnectionError, match="graph outage"):
        reconciler.prune_graph_projection_generations(
            projection_id=row.id, page_size=1, retain=1, dry_run=False
        )
    row.refresh_from_db()
    assert row.pruned_at is None
    assert row.state == "superseded"
    graph.delete_generation = lambda **kwargs: False  # Already absent is completed.
    reconciler.prune_graph_projection_generations(
        projection_id=row.id, page_size=1, retain=1, dry_run=False
    )
    row.refresh_from_db()
    assert row.pruned_at is not None


@pytest.mark.parametrize("prune_first", [False, True])
def test_prune_and_worker_claim_are_exclusive_in_both_orderings(authority, prune_first):
    _collection, create = authority
    row = create("failed")
    repository = FunctionProjectionStateRepository()
    now = timezone.now()

    def begin():
        return repository.begin_prune(
            projection_id=row.id, generation_key=row.generation_key, now=now
        )

    def claim():
        return repository.claim(
            projection_id=row.id, owner="audit-worker", now=now, lease_seconds=60
        )

    if prune_first:
        assert begin() is True
        assert claim() is None
    else:
        assert claim() is not None
        assert begin() is False
    row.refresh_from_db()
    assert row.state == ("superseded" if prune_first else "building")
    assert row.pruned_at is None


@pytest.mark.parametrize("wrong_field", ["projection_id", "generation_key"])
def test_pruning_requires_exact_projection_and_generation(authority, wrong_field):
    _collection, create = authority
    row = create("failed")
    repository = FunctionProjectionStateRepository()
    identity = {"projection_id": row.id, "generation_key": row.generation_key}
    wrong = {**identity, wrong_field: uuid4()}
    assert repository.begin_prune(**wrong, now=timezone.now()) is False
    row.refresh_from_db()
    assert row.state == "failed"
    assert repository.begin_prune(**identity, now=timezone.now()) is True
    assert repository.record_pruned(**wrong, now=timezone.now()) is False
    row.refresh_from_db()
    assert row.pruned_at is None
    assert repository.record_pruned(**identity, now=timezone.now()) is True
    assert repository.record_pruned(**identity, now=timezone.now()) is False


def test_state_role_can_execute_prune_functions_but_cannot_update_table(authority):
    _collection, create = authority
    row = create("failed")
    repository = FunctionProjectionStateRepository(state_using="default")
    with transaction.atomic():
        with connection.cursor() as cursor:
            cursor.execute("SET LOCAL ROLE aquillm_projection_state")
        assert repository.begin_prune(
            projection_id=row.id, generation_key=row.generation_key, now=timezone.now()
        )
        assert repository.record_pruned(
            projection_id=row.id, generation_key=row.generation_key, now=timezone.now()
        )
        with pytest.raises(ProgrammingError, match="permission denied"):
            with transaction.atomic(), connection.cursor() as cursor:
                cursor.execute(
                    "UPDATE public.apps_knowledge_graph_collectiongraphprojection "
                    "SET pruned_at = NULL WHERE id = %s",
                    [row.id],
                )
    row.refresh_from_db()
    assert row.pruned_at is not None


def test_pruned_generations_are_omitted_from_materialized_authority(authority):
    collection, create = authority
    completed, retained = create(), create()
    repository = FunctionProjectionStateRepository()
    assert repository.begin_prune(
        projection_id=completed.id,
        generation_key=completed.generation_key,
        now=timezone.now(),
    )
    assert repository.record_pruned(
        projection_id=completed.id,
        generation_key=completed.generation_key,
        now=timezone.now(),
    )
    page = generation_audit._projection_page(
        after_id=None, page_size=10, collection_id=collection.pk
    )
    assert [row.id for row in page] == [retained.id]
    assert CollectionGraphProjection.objects.filter(pk=completed.id).exists()
