"""Shared setup and event probes for the focused scalar overlap tests."""
from contextlib import contextmanager
from threading import Event
from types import SimpleNamespace


@contextmanager
def existing_query(legacy, alias="default", query="source"):
    from apps.documents.models import TextChunk
    from apps.documents.services.chunk_search_lifetime import enter_search
    from apps.documents.services.chunk_search_query_observation import query_embedding
    from apps.documents.services.source_loading import source_runtime_scope
    from apps.documents.tests.test_embedding_generation_query import runtime_for
    from lib.retrieval.operation import operation_scope
    runtime, policy = runtime_for(legacy)
    if alias != "default":
        from apps.collections.services.retrieval_authorization import (
            bind_retrieval_reauthorization_capability, freeze_retrieval_authorization_context,
        )
        principal = object()
        runtime.authorization = freeze_retrieval_authorization_context(
            principal=principal, database_alias=alias, policy=policy,
            selected_collection_ids=(legacy[1].collection_id,),
            selected_document_ids=(legacy[1].id,),
            reauthorization_capability=bind_retrieval_reauthorization_capability(
                principal=principal, policy=policy))
    with source_runtime_scope(runtime), operation_scope(runtime.budget):
        lifetime, scope = enter_search(TextChunk, (legacy[0],), runtime.authorization, query, True)
        with lifetime:
            yield runtime, scope, query_embedding(query)


def signature(snapshot):
    return (
        snapshot.vector_chunk_ids, snapshot.trigram_chunk_ids, snapshot.exact_chunk_ids,
        snapshot.full_text_chunk_ids, snapshot.source_admission.channels,
        snapshot.source_admission.initial_ids, snapshot.source_admission.graph_ceiling,
        snapshot.graph_seeds, tuple(r.pk for r in snapshot.baseline_candidates),
        tuple(c.protected_exact for c in snapshot.fusion_v2.candidates),
    )


def install_overlap_probe(monkeypatch, runtime, scope):
    from threading import get_ident
    from django.db import connections
    from apps.documents.services import embedding_index_query
    from apps.documents.services import graph_candidate_metadata as metadata
    from apps.documents.services import graph_metadata_worker as worker
    lexical_entered, dense_entered = Event(), Event()
    observed = {"parent_thread": get_ident()}
    original_trigram, original_exact = metadata._trigram_refs, metadata._exact_refs
    original_dense = embedding_index_query.rank_compatible_chunks
    def trigram(*args, **kwargs):
        acquired = runtime.lock.acquire(blocking=False)
        if acquired:
            runtime.lock.release()
        connection = connections[scope.using]
        observed.update(alias=scope.using, thread=get_ident(), parent_lock_retained=not acquired,
                        connection=connection, trigram_profile=worker._profile(connection))
        lexical_entered.set()
        assert dense_entered.wait(3)
        return original_trigram(*args, **kwargs)
    def exact(*args, **kwargs):
        connection = connections[scope.using]
        observed.update(exact_connection=connection, exact_profile=worker._profile(connection))
        refs, terms = original_exact(*args, **kwargs)
        assert refs and terms  # This reader actually evaluated nonempty exact SQL.
        return refs, terms
    def dense(*args, **kwargs):
        dense_entered.set()
        assert lexical_entered.wait(3)
        return original_dense(*args, **kwargs)
    monkeypatch.setattr(metadata, "_trigram_refs", trigram)
    monkeypatch.setattr(metadata, "_exact_refs", exact)
    monkeypatch.setattr(embedding_index_query, "rank_compatible_chunks", dense)
    return observed


def assert_between_lane_rejection(monkeypatch, legacy, runtime, scope, observation, query, failure):
    import pytest
    from apps.documents.services import embedding_index_query
    from apps.documents.services import graph_candidate_metadata as metadata
    from apps.documents.services import graph_metadata_worker as worker
    from apps.documents.tests.test_graph_admission_channels import capture
    entered, release, exact_dispatched = Event(), Event(), Event()
    original_trigram, original_dense = metadata._trigram_refs, embedding_index_query.rank_compatible_chunks
    original_start, jobs = worker.start_lexical_metadata, []
    def start(*args, **kwargs):
        job = original_start(*args, **kwargs)
        jobs.append(job)
        return job
    def trigram(*args, **kwargs):
        refs = original_trigram(*args, **kwargs)
        assert refs  # First lexical SQL really completed before invalidation.
        entered.set()
        assert release.wait(3)
        return refs
    def exact(*args, **kwargs):
        exact_dispatched.set()
        raise AssertionError("exact SQL dispatched after between-lane invalidation")
    def dense(*args, **kwargs):
        assert entered.wait(3)
        try:
            return original_dense(*args, **kwargs)
        finally:
            if failure == "invalidate":
                jobs[0].invalidate()
            else:
                jobs[0].deadline = worker.monotonic() - 1
            release.set()
    monkeypatch.setattr(worker, "start_lexical_metadata", start)
    monkeypatch.setattr(metadata, "_trigram_refs", trigram)
    monkeypatch.setattr(metadata, "_exact_refs", exact)
    monkeypatch.setattr(embedding_index_query, "rank_compatible_chunks", dense)
    try:
        with pytest.raises(ValueError):
            capture(legacy, runtime, query, observation)
    finally:
        release.set()
        assert len(jobs) == 1 and jobs[0] is not None
        with pytest.raises(ValueError):
            jobs[0]._future.result(timeout=3)
    assert not exact_dispatched.is_set()
    assert runtime.budget.sources_used == 0
    assert runtime.cache == scope.bodies == scope.dense_rows == {}
    assert scope.fusion_v2 is None


def assert_delayed_entry_rejected(monkeypatch, worker, failure):
    """Normal dispatch is delayed before _read; discarded work cannot set up DB."""
    import sys
    import pytest
    entered, release, cleanup, cleanup_release = Event(), Event(), Event(), Event()
    counts = dict(connection=0, setup=0, data=0)
    def data(*args):
        counts["data"] += 1
    @contextmanager
    def boundary(*args):
        counts["setup"] += 1
        yield
    class Connections:
        def __getitem__(self, alias):
            counts["connection"] += 1
            return SimpleNamespace(execute_wrappers=[])
        def close_all(self):
            cleanup.set()
            assert cleanup_release.wait(3)
    monkeypatch.setattr(worker, "connections", Connections())
    monkeypatch.setitem(sys.modules, "apps.documents.services.source_loading",
                        SimpleNamespace(bounded_source_database=boundary))
    monkeypatch.setitem(sys.modules, "apps.documents.services.graph_candidate_metadata",
                        SimpleNamespace(_lexical_text=data, _trigram_refs=data, _exact_refs=data))
    original = worker.LexicalMetadataJob._read
    def delayed(self):
        entered.set()
        assert release.wait(3)
        return original(self)
    monkeypatch.setattr(worker.LexicalMetadataJob, "_read", delayed)
    monkeypatch.setattr(worker.LexicalMetadataJob, "_original_remaining", lambda self: 5000)
    monkeypatch.setattr(worker, "_failure", lambda: ValueError("discarded"))
    job = worker.start_lexical_metadata(SimpleNamespace(alias="original"),
        parent_profile=None, runtime=None, budget=None, check=lambda: None)
    assert entered.wait(3)
    if failure == "invalidate":
        job.invalidate()
    else:
        job.deadline = worker.monotonic() - 1
    try:
        assert not worker._SLOT.acquire(blocking=False)
        release.set()
        assert cleanup.wait(3)
        assert not job._future.done()
        assert not worker._SLOT.acquire(blocking=False)
    finally:
        release.set()
        cleanup_release.set()
    with pytest.raises(ValueError, match="discarded"):
        job._future.result(timeout=3)
    assert counts == dict(connection=0, setup=0, data=0)
    assert worker._SLOT.acquire(blocking=False)
    worker._SLOT.release()
