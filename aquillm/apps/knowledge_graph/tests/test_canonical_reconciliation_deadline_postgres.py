"""Real PostgreSQL connections reproduce contention and enforce total fences."""

from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from threading import Event
from time import monotonic, sleep

import pytest
from django.db import connection, connections, transaction
from django.utils import timezone

from apps.collections.models import Collection
from apps.knowledge_graph.tests.test_models import _artifact

pytestmark = pytest.mark.django_db(transaction=True)


def fixture_artifact():
    collection = Collection.objects.create(name="G5 synthetic contention")
    artifact = _artifact(
        scope_type="collection", scope_id=collection.pk, status="active"
    )
    artifact.save()
    return collection, artifact


def other_connection_publication(projection_id):
    from apps.knowledge_graph.projection.lifecycle import (
        publish_projection_ready_compare_and_set,
    )
    from apps.knowledge_graph.projection.memgraph_repository import (
        ProjectionValidationV1,
    )
    from apps.knowledge_graph.projection.records import ProjectionCountsV1

    started = monotonic()
    try:
        outcome = publish_projection_ready_compare_and_set(
            projection_id=projection_id,
            owner="g5",
            using="default",
            now=timezone.now(),
            expected_generation_key="a" * 64,
            expected_graph_checksum="a" * 64,
            expected_private_mapping_checksum="c" * 64,
            validation=ProjectionValidationV1(
                "a" * 64, "a" * 64, ProjectionCountsV1(0, 0, 0, 0, 0, 0, 0, 0, 0), True
            ),
        )
        return monotonic() - started, outcome.published
    finally:
        connections.close_all()


def test_legacy_barrier_blocks_publication_but_v2_compute_does_not(monkeypatch):
    from apps.knowledge_graph.resolution import canonical
    from apps.knowledge_graph.resolution import canonical_reconciliation_v2 as v2
    from apps.knowledge_graph.tests.ready_subset_support import ScopeFixture

    fixture = ScopeFixture(monkeypatch)
    projection = fixture.persist_ready("A")
    monkeypatch.setattr(
        "apps.knowledge_graph.projection.lifecycle._projection_versions",
        lambda: ("collection-graph-v1", "projection-v1", "key-v1"),
    )

    def reset_projection():
        projection.state, projection.ready_at = "building", None
        projection.lease_owner, projection.lease_expires_at = (
            "g5",
            timezone.now() + timedelta(seconds=30),
        )
        projection.save()

    reset_projection()
    with ThreadPoolExecutor(max_workers=1) as pool:
        real = canonical.resolve_canonical_entities
        blocked = []

        def locked_compute(*args, **kwargs):
            assert connection.in_atomic_block
            blocked.append(pool.submit(other_connection_publication, projection.pk))
            sleep(0.2)
            assert not blocked[0].done()
            return real(*args, **kwargs)

        monkeypatch.setattr(canonical, "resolve_canonical_entities", locked_compute)
        canonical.rebuild_canonical_registry(using="default")
        elapsed, published = blocked[0].result(timeout=2)
        assert elapsed >= 0.18 and published
        reset_projection()
        observed = []

        def compute(*args, **kwargs):
            assert not connection.in_atomic_block
            publication = pool.submit(other_connection_publication, projection.pk)
            sleep(0.2)
            observed.append(publication.result(timeout=1))
            return real(*args, **kwargs)

        monkeypatch.setattr(canonical, "resolve_canonical_entities", compute)
        result = v2.reconcile_canonical_v2(
            using="default", deadline=monotonic() + 2, clock=monotonic
        )
    assert result.status == "published"
    assert observed[0][0] < 0.18 and observed[0][1]


def test_source_changes_during_compute_return_stale_without_retry(monkeypatch):
    from apps.knowledge_graph.resolution import canonical
    from apps.knowledge_graph.resolution import canonical_reconciliation_v2 as v2

    collection, artifact = fixture_artifact()
    real = canonical.resolve_canonical_entities
    calls = []

    def compute(*args, **kwargs):
        calls.append(1)
        artifact.status = "superseded"
        artifact.save(update_fields=["status"])
        return real(*args, **kwargs)

    monkeypatch.setattr(canonical, "resolve_canonical_entities", compute)
    outcome = v2.reconcile_canonical_v2(
        using="default", deadline=monotonic() + 2, clock=monotonic
    )
    assert outcome.status == "stale"
    assert calls == [1]


def test_expired_compute_does_not_publish(monkeypatch):
    from apps.knowledge_graph.resolution import canonical
    from apps.knowledge_graph.resolution import canonical_reconciliation_v2 as v2

    fixture_artifact()
    real = canonical.resolve_canonical_entities

    def compute(*args, **kwargs):
        sleep(0.2)
        return real(*args, **kwargs)

    monkeypatch.setattr(canonical, "resolve_canonical_entities", compute)
    outcome = v2.reconcile_canonical_v2(
        using="default", deadline=monotonic() + 0.15, clock=monotonic
    )
    assert outcome.status == "timed_out"


def test_lock_wait_obeys_total_deadline_and_stricter_operator_limit():
    from apps.knowledge_graph.resolution import canonical_reconciliation_v2 as v2

    collection, artifact = fixture_artifact()
    acquired, release = Event(), Event()

    def hold():
        try:
            with transaction.atomic():
                Collection.objects.select_for_update().get(pk=collection.pk)
                acquired.set()
                assert release.wait(3)
        finally:
            connections.close_all()

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(hold)
        assert acquired.wait(2)
        try:
            with connection.cursor() as cursor:
                cursor.execute("SET lock_timeout = '40ms'")
            start = monotonic()
            result = v2.reconcile_canonical_v2(
                using="default", deadline=start + 0.7, clock=monotonic
            )
            elapsed = monotonic() - start
            assert result.status == "timed_out"
            assert elapsed < 0.5
        finally:
            with connection.cursor() as cursor:
                cursor.execute("SET lock_timeout = '0'")
            release.set()
        future.result(timeout=2)
