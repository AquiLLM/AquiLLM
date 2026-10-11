"""Nonempty provenance capture must wait at the lifecycle collection prefix."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from time import monotonic, sleep

import pytest
from django.apps import apps
from django.db import connection, connections

from apps.documents.models import TextChunk
from apps.knowledge_graph.graph import invalidation as lifecycle
from apps.knowledge_graph.models import CollectionEntityDocumentLink, GraphArtifact
from apps.knowledge_graph.resolution import canonical
from apps.knowledge_graph.resolution.canonical_reconciliation_v2 import (
    reconcile_canonical_v2,
)
from apps.knowledge_graph.tests.ready_subset_support import ScopeFixture

pytestmark = pytest.mark.django_db(transaction=True)


def retained_source(reference):
    owner = apps.get_model(reference.model_label).objects.filter(
        pkid=reference.pkid, id=reference.id
    ).values_list("id", "full_text", "full_text_hash", "collection_id").get()
    chunks = tuple(TextChunk.objects.filter(doc_id=reference.id).order_by("pk").values_list(
        "pk", "doc_id", "chunk_number", "start_position", "end_position", "content", "representation_generation_id",
        "legacy_source_snapshot_id"))
    assert chunks
    return owner, chunks


def test_initial_capture_waits_before_provenance_while_real_cleanup_holds_prefix(
    monkeypatch,
):
    fixture = ScopeFixture(monkeypatch)
    projection = fixture.persist_ready("A", with_evidence=True)
    reference = fixture.snapshot.selections[0].document
    retained = retained_source(reference)
    document = lifecycle.DocumentLifecycleRef(
        reference.model_label, reference.pkid, reference.id
    )
    assert (
        CollectionEntityDocumentLink.objects.filter(
            artifact_id=projection.artifact_id
        ).count()
        == 1
    )
    cleanup_locked, release, capture_started, provenance_entered = (
        Event(),
        Event(),
        Event(),
        Event(),
    )
    pids = {}
    original_phases = lifecycle._lock_document_graph_phases
    original_inputs = canonical._load_locked_canonical_inputs

    def phases(*args, **kwargs):
        result = original_phases(*args, **kwargs)
        with connections["default"].cursor() as cursor:
            cursor.execute("SELECT pg_backend_pid()")
            pids["cleanup"] = cursor.fetchone()[0]
        cleanup_locked.set()
        assert release.wait(5)
        return result

    def inputs(**kwargs):
        provenance_entered.set()
        return original_inputs(**kwargs)

    monkeypatch.setattr(lifecycle, "_lock_document_graph_phases", phases)
    monkeypatch.setattr(canonical, "_load_locked_canonical_inputs", inputs)

    def cleanup():
        try:
            return lifecycle.cleanup_document_graph_state(
                document,
                (7,),
                reason="document_content_changed",
                using="default",
                expected_source_hash=reference.source_hash,
                fail_on_stale=True,
            )
        finally:
            connections.close_all()

    def capture():
        try:
            with connections["default"].cursor() as cursor:
                cursor.execute("SELECT pg_backend_pid()")
                pids["capture"] = cursor.fetchone()[0]
            capture_started.set()
            return reconcile_canonical_v2(
                using="default", deadline=monotonic() + 2, clock=monotonic
            )
        finally:
            connections.close_all()

    with ThreadPoolExecutor(max_workers=2) as pool:
        cleanup_future = pool.submit(cleanup)
        assert cleanup_locked.wait(2)
        capture_future = pool.submit(capture)
        try:
            assert capture_started.wait(2)
            limit, blocked = monotonic() + 1, False
            while monotonic() < limit:
                with connection.cursor() as cursor:
                    cursor.execute("SELECT pg_blocking_pids(%s)", [pids["capture"]])
                    blocked = pids["cleanup"] in cursor.fetchone()[0]
                if blocked:
                    break
                sleep(0.01)
            assert blocked, "capture must actually wait on the real cleanup transaction"
            assert not provenance_entered.is_set(), (
                "capture locked provenance before collection prefix"
            )
        finally:
            release.set()
        cleanup_future.result(timeout=4)
        outcome = capture_future.result(timeout=4)
    assert outcome.status == "stale"
    assert not GraphArtifact.objects.filter(pk=projection.artifact_id).exists()
    assert not CollectionEntityDocumentLink.objects.exists()
    assert retained_source(reference) == retained


def test_nonempty_capture_releases_locks_before_real_cleanup_during_compute(
    monkeypatch,
):
    fixture = ScopeFixture(monkeypatch)
    fixture.persist_ready("A", with_evidence=True)
    reference = fixture.snapshot.selections[0].document
    retained = retained_source(reference)
    document = lifecycle.DocumentLifecycleRef(
        reference.model_label, reference.pkid, reference.id
    )
    real_compute, calls = canonical.resolve_canonical_entities, []

    def cleanup():
        try:
            return lifecycle.cleanup_document_graph_state(
                document,
                (7,),
                reason="document_content_changed",
                using="default",
                expected_source_hash=reference.source_hash,
                fail_on_stale=True,
            )
        finally:
            connections.close_all()

    with ThreadPoolExecutor(max_workers=1) as pool:

        def compute(inputs, **kwargs):
            assert not connection.in_atomic_block
            assert len(inputs) == 1 and CollectionEntityDocumentLink.objects.exists()
            calls.append(1)
            pool.submit(cleanup).result(timeout=2)
            assert not CollectionEntityDocumentLink.objects.exists()
            return real_compute(inputs, **kwargs)

        monkeypatch.setattr(canonical, "resolve_canonical_entities", compute)
        outcome = reconcile_canonical_v2(
            using="default", deadline=monotonic() + 3, clock=monotonic
        )
    assert outcome.status == "stale" and calls == [1]
    assert retained_source(reference) == retained
