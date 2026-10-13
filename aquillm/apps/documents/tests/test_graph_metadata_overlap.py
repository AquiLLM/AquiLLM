"""Committed PostgreSQL fixtures exercise native scalar overlap and session fences."""
from threading import Event, get_ident

import pytest
from django.conf import settings
from django.db import connections, transaction

from apps.documents.services import graph_candidate_metadata as metadata
from apps.documents.services import graph_metadata_worker as worker
from apps.documents.tests.test_graph_admission_channels import capture
from apps.documents.tests.test_graph_admission_query_binding import serving_query
from apps.documents.tests.test_embedding_generation import legacy as legacy
from apps.documents.tests.lexical_owner_support import unchunked as unchunked
from lib.embeddings.tests.test_document_provider import transport as transport

from apps.documents.tests.graph_metadata_test_support import existing_query as _existing_query, signature as _signature

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.mark.django_db(transaction=True, databases=["default", "projection_source"])
@pytest.mark.parametrize("alias", ["default", "projection_source"])
def test_native_overlap_preserves_ordered_channels_and_retains_parent_lock(
    unchunked, transport, monkeypatch, alias,
):
    from apps.documents.tests.graph_metadata_test_support import install_overlap_probe
    from apps.documents.tests.test_lexical_relevance import replace_chunks
    from dataclasses import replace
    legacy = unchunked
    rows = replace_chunks(legacy, ["ATP distractor"] * 30 +
                          ["ATP BRCA1 answer", "ATP BRCA1 second"])
    legacy = (legacy[0], replace(legacy[1], source_hash=legacy[0].full_text_hash))
    monkeypatch.setattr(settings, "RAG_FULL_TEXT_ENABLED", False, raising=False)
    monkeypatch.setattr(settings, "RAG_CANDIDATE_MULTIPLIER", 4.0, raising=False)
    monkeypatch.setattr(settings, "RAG_SHORT_QUERY_CANDIDATE_SCALE", 1.0, raising=False)
    monkeypatch.setattr(settings, "RAG_RRF_ENABLED", True, raising=False)
    monkeypatch.setattr(settings, "RAG_PROTECTED_EXACT_ENABLED", True, raising=False)
    query = '"ATP BRCA1" ATP'
    start = worker.start_lexical_metadata
    with serving_query(legacy, transport, monkeypatch):
        pass  # Commit the real dense generation before either independent original turn.
    with _existing_query(legacy, query=query) as (runtime, scope, observation):
        monkeypatch.setattr(worker, "start_lexical_metadata", lambda *a, **kw: None)
        serial = capture(legacy, runtime, query, observation)
    monkeypatch.setattr(worker, "start_lexical_metadata", start)
    with _existing_query(legacy, alias, query) as (runtime, scope, observation):
        observed = install_overlap_probe(monkeypatch, runtime, scope)
        with connections[scope.using].cursor() as cursor:
            cursor.execute("SELECT pg_backend_pid()")
            parent_pid = cursor.fetchone()[0]
        overlap = capture(legacy, runtime, query, observation)
        assert observed["thread"] != observed["parent_thread"]
        assert observed["alias"] == alias
        trigram, exact = observed["trigram_profile"], observed["exact_profile"]
        assert trigram.pid == exact.pid != parent_pid
        assert observed["connection"] is observed["exact_connection"]
        assert 0 < exact.statement_ms <= trigram.statement_ms
        assert 0 < exact.lock_ms <= trigram.lock_ms
        assert observed["parent_lock_retained"]
        refs = dict(overlap.source_admission.channels)["exact"]
        assert tuple(r.pk for r in refs) == tuple(r.pk for r in rows[-2:] + rows[:-2])
        assert tuple(r.exact_coverage for r in refs) == (2, 2) + (1,) * 30
        assert rows[-2].pk in overlap.source_admission.initial_ids
        assert any(c.protected_exact for c in overlap.fusion_v2.candidates)
        assert _signature(overlap) == _signature(serial)
        assert runtime.budget.sources_used == len(serial.baseline_candidates) == 25
        assert overlap.source_admission.graph_ceiling == 20


@pytest.mark.parametrize("failure", ["invalidate", "physical_expiry"])
def test_completed_trigram_cannot_dispatch_exact_after_invalidation(
    legacy, transport, monkeypatch, failure,
):
    from apps.documents.tests.graph_metadata_test_support import assert_between_lane_rejection
    monkeypatch.setattr(settings, "RAG_FULL_TEXT_ENABLED", False, raising=False)
    with serving_query(legacy, transport, monkeypatch):
        pass
    query = '"source"'
    with _existing_query(legacy, query=query) as (runtime, scope, observation):
        assert_between_lane_rejection(monkeypatch, legacy, runtime, scope, observation, query, failure)


@pytest.mark.parametrize("transaction_mode", ["atomic", "manual"])
def test_caller_transaction_retains_serial_visibility(legacy, transport, monkeypatch, transaction_mode):
    from apps.documents.models import TextChunk
    monkeypatch.setattr(settings, "RAG_FULL_TEXT_ENABLED", False, raising=False)
    def unexpected(*args, **kwargs):
        raise AssertionError("worker dispatched inside caller transaction")
    monkeypatch.setattr(worker, "start_lexical_metadata", unexpected)
    with serving_query(legacy, transport, monkeypatch) as (runtime, scope, observation):
        connection = connections[scope.using]
        if transaction_mode == "manual":
            connection.set_autocommit(False)
        try:
            with transaction.atomic(using=scope.using):
                TextChunk.objects.using(scope.using).filter(doc_id=legacy[0].id).update(
                    content="source modified inside caller transaction")
                result = capture(legacy, runtime, "source", observation)
                assert result.trigram_chunk_ids and result.baseline_candidates
                transaction.set_rollback(True, using=scope.using)
        finally:
            if transaction_mode == "manual":
                connection.rollback()
                connection.set_autocommit(True)


@pytest.mark.parametrize("statement,lock,worker_statement,worker_lock", [
    (30, 9, 0, 0), (30, 9, 20, 3), (0, 0, 0, 0)])
def test_worker_installs_parent_and_remaining_timeout_minima(
    legacy, transport, monkeypatch, statement, lock, worker_statement, worker_lock,
):
    from django.db.backends.signals import connection_created
    from threading import current_thread
    monkeypatch.setattr(settings, "RAG_FULL_TEXT_ENABLED", False, raising=False)
    observed = []
    original = metadata._trigram_refs
    def guard(sender, connection, **kwargs):
        if current_thread().name == "source-lexical-metadata":
            with connection.cursor() as cursor:
                cursor.execute("SELECT set_config('statement_timeout', %s, false), "
                    "set_config('lock_timeout', %s, false)",
                    [f"{worker_statement}ms", f"{worker_lock}ms"])
    def inspect(*args, **kwargs):
        profile = worker._profile(connections["default"])
        observed.append(profile)
        return original(*args, **kwargs)
    with serving_query(legacy, transport, monkeypatch) as (runtime, scope, observation):
        connection = connections[scope.using]
        with connection.cursor() as cursor:
            cursor.execute("SELECT set_config('statement_timeout', %s, false), "
                           "set_config('lock_timeout', %s, false)",
                           [f"{statement}ms", f"{lock}ms"])
        monkeypatch.setattr(metadata, "_trigram_refs", inspect)
        connection_created.connect(guard, weak=False)
        try:
            capture(legacy, runtime, "source", observation)
            assert len(observed) == 1
            assert 0 < observed[0].statement_ms <= (statement or runtime.budget.limits.retrieval_ms)
            assert 0 < observed[0].lock_ms <= (lock or runtime.budget.limits.retrieval_ms)
            assert observed[0].statement_ms <= (worker_statement or runtime.budget.limits.retrieval_ms)
            assert observed[0].lock_ms <= (worker_lock or runtime.budget.limits.retrieval_ms)
        finally:
            connection_created.disconnect(guard)
            with connection.cursor() as cursor:
                cursor.execute("RESET statement_timeout")
                cursor.execute("RESET lock_timeout")


@pytest.mark.parametrize("setting,value", [
    ("default_transaction_isolation", "repeatable read"),
    ("default_transaction_read_only", "on"),
    ("row_security", "off"), ("search_path", "pg_catalog, public"),
])
def test_successful_worker_policy_mismatch_cleans_up_before_serial_fallback(
    legacy, transport, monkeypatch, setting, value,
):
    from django.db.backends.signals import connection_created
    from threading import current_thread
    from django.db.backends.base.base import BaseDatabaseWrapper
    monkeypatch.setattr(settings, "RAG_FULL_TEXT_ENABLED", False, raising=False)
    parent_thread = get_ident()
    observed = {"worker_closed": False, "fresh_guard": False}
    original_close = BaseDatabaseWrapper.close
    original_trigram = metadata._trigram_refs
    def guard(sender, connection, **kwargs):
        if current_thread().name == "source-lexical-metadata":
            with connection.cursor() as cursor:
                cursor.execute("SELECT set_config(%s, %s, false)", [setting, value])
            observed["fresh_guard"] = True
    def close(connection):
        result = original_close(connection)
        if current_thread().name == "source-lexical-metadata":
            observed["worker_closed"] = True
        return result
    def trigram(*args, **kwargs):
        assert get_ident() == parent_thread
        assert observed["worker_closed"]
        return original_trigram(*args, **kwargs)
    monkeypatch.setattr(BaseDatabaseWrapper, "close", close)
    monkeypatch.setattr(metadata, "_trigram_refs", trigram)
    connection_created.connect(guard, weak=False)
    try:
        with serving_query(legacy, transport, monkeypatch) as (runtime, scope, observation):
            snapshot = capture(legacy, runtime, "source", observation)
            assert snapshot.trigram_chunk_ids and snapshot.baseline_candidates
            assert observed == {"worker_closed": True, "fresh_guard": True}
    finally:
        connection_created.disconnect(guard)


@pytest.mark.parametrize("mutation", ["query", "operation", "closed"])
def test_original_owner_mutation_rejects_before_body_publication(
    legacy, transport, monkeypatch, mutation,
):
    from apps.documents.services import embedding_index_query
    from apps.documents.services.representation_contracts import SourceScopeUnavailable
    from lib.retrieval.operation import Operation, _OPERATION
    from asyncio import CancelledError
    monkeypatch.setattr(settings, "RAG_FULL_TEXT_ENABLED", False, raising=False)
    with serving_query(legacy, transport, monkeypatch) as (runtime, scope, observation):
        entered, release, cleaned = Event(), Event(), Event()
        original_trigram = metadata._trigram_refs
        original_dense = embedding_index_query.rank_compatible_chunks
        original_cleanup = worker.connections.close_all
        original_start, jobs = worker.start_lexical_metadata, []
        token = None
        def start(*args, **kwargs):
            job = original_start(*args, **kwargs)
            jobs.append(job)
            return job
        def trigram(*args, **kwargs):
            entered.set()
            assert release.wait(3)
            return original_trigram(*args, **kwargs)
        def cleanup():
            original_cleanup()
            cleaned.set()
        def dense(*args, **kwargs):
            nonlocal token
            assert entered.wait(3)
            if mutation == "query":
                scope.query_digest = "0" * 64
            elif mutation == "operation":
                token = _OPERATION.set(Operation(runtime.budget))
            else:
                runtime.budget.close("test owner closed")
            release.set()
            return original_dense(*args, **kwargs)
        monkeypatch.setattr(metadata, "_trigram_refs", trigram)
        monkeypatch.setattr(embedding_index_query, "rank_compatible_chunks", dense)
        monkeypatch.setattr(worker.connections, "close_all", cleanup)
        monkeypatch.setattr(worker, "start_lexical_metadata", start)
        try:
            with pytest.raises((SourceScopeUnavailable, ValueError, CancelledError)):
                capture(legacy, runtime, "source", observation)
            assert runtime.cache == scope.bodies == scope.dense_rows == {}
            assert scope.fusion_v2 is None
            assert runtime.budget.sources_used == 0
        finally:
            release.set()
            assert cleaned.wait(3)
            assert len(jobs) == 1 and jobs[0] is not None
            try:
                jobs[0]._future.result(timeout=3)
            except BaseException:
                assert jobs[0]._future.done()
            if token is not None:
                _OPERATION.reset(token)


@pytest.mark.parametrize("lane", ["dense", "lexical"])
def test_lexical_failure_propagates_while_dense_failure_keeps_lexical_results(
    legacy, transport, monkeypatch, lane,
):
    from apps.documents.services import embedding_index_query
    monkeypatch.setattr(settings, "RAG_FULL_TEXT_ENABLED", False, raising=False)
    with serving_query(legacy, transport, monkeypatch) as (runtime, scope, observation):
        def fail(*args, **kwargs):
            raise RuntimeError("selected metadata lane failed")
        if lane == "lexical":
            monkeypatch.setattr(metadata, "_trigram_refs", fail)
            with pytest.raises(RuntimeError, match="selected metadata lane failed"):
                capture(legacy, runtime, "source", observation)
            assert runtime.budget.sources_used == 0
            assert runtime.cache == scope.bodies == {}
        else:
            monkeypatch.setattr(embedding_index_query, "rank_compatible_chunks", fail)
            snapshot = capture(legacy, runtime, "source", observation)
            assert snapshot.vector_error == "upstream_unavailable"
            assert snapshot.vector_chunk_ids == ()
            assert snapshot.trigram_chunk_ids and snapshot.baseline_candidates
