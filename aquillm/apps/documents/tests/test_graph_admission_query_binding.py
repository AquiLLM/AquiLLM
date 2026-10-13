"""Native admission consumes the actual serving query-preparation lifecycle."""

from contextlib import contextmanager
from dataclasses import replace

import pytest
from django.conf import settings

from apps.documents.models import TextChunk
from apps.documents.services.chunk_search_lifetime import enter_search
from apps.documents.services.chunk_search_query_observation import query_embedding
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.services.source_loading import source_runtime_scope
from apps.documents.tests.test_embedding_generation import legacy as _legacy_fixture, stage
from apps.documents.tests.test_embedding_generation_query import runtime_for
from apps.documents.tests.test_graph_admission_channels import capture
from lib.embeddings.document_contracts import digest
from lib.embeddings.tests.test_document_provider import transport as _transport_fixture
from lib.retrieval.operation import operation_scope

legacy = _legacy_fixture
transport = _transport_fixture
pytestmark = pytest.mark.django_db(transaction=True)


@contextmanager
def serving_query(legacy, transport, monkeypatch, *, unavailable=False):
    from apps.documents.services import embedding_index_store as store

    _, lease, _, _ = stage(legacy, transport)
    assert store.activate_embedding_generation(lease, using="default") == "activated"
    monkeypatch.setattr(settings, "DOCUMENT_EMBEDDING_SPACE_KEY", transport[0].space_key)
    monkeypatch.setattr(settings, "RAG_CACHE_ENABLED", False, raising=False)
    runtime, _ = runtime_for(legacy)
    with source_runtime_scope(runtime), operation_scope(runtime.budget):
        lifetime, scope = enter_search(TextChunk, (legacy[0],), runtime.authorization, "source", True)
        with lifetime:
            unprepared = scope.query_digest
            if unavailable:
                from apps.documents.services import chunk_search_query_observation as queries

                def fail(*args, **kwargs):
                    raise RuntimeError("synthetic verified transport outage")

                monkeypatch.setattr(queries, "observe_query_embedding", fail)
                with pytest.raises(RuntimeError):
                    query_embedding("source")
                observation = None
                assert scope.query_digest != unprepared
            else:
                observation = query_embedding("source")
                assert scope.query_digest == observation.prepared_input_sha256 != unprepared
            yield runtime, scope, observation


def test_prepared_serving_query_admits_actual_native_dense_and_lexical_rows(legacy, transport, monkeypatch):
    with serving_query(legacy, transport, monkeypatch) as (runtime, scope, observation):
        snapshot = capture(legacy, runtime, "source", observation)
        assert snapshot.source_admission is not None
        assert snapshot.vector_chunk_ids and snapshot.baseline_candidates
        assert runtime.budget.sources_used == len(snapshot.baseline_candidates) == 1
        assert scope.query_digest == observation.prepared_input_sha256
        snapshot.source_admission.check()


def test_prepared_digest_after_provider_failure_preserves_lexical_fallback(legacy, transport, monkeypatch):
    with serving_query(legacy, transport, monkeypatch, unavailable=True) as (runtime, scope, observation):
        assert observation is None
        prepared = scope.query_digest
        snapshot = capture(legacy, runtime, "source", observation)
        assert snapshot.source_admission is not None and not snapshot.vector_chunk_ids
        assert snapshot.trigram_chunk_ids and snapshot.baseline_candidates
        assert runtime.budget.sources_used == 1 and scope.query_digest == prepared


@pytest.mark.parametrize("mutation", ["query", "digest", "prepared_to_unprepared", "compatibility"])
def test_prepared_query_binding_drift_rejects_before_body_publication(legacy, transport, monkeypatch, mutation):
    with serving_query(legacy, transport, monkeypatch) as (runtime, scope, observation):
        query = "source"
        if mutation == "query":
            query = "source revised"
        elif mutation == "digest":
            scope.query_digest = digest({"version": "unprepared-query-v1", "query": "wrong query"})
        elif mutation == "prepared_to_unprepared":
            scope.query_digest = digest({"version": "unprepared-query-v1", "query": query})
        else:
            scope.query_compatibility = "b" * 64
        with pytest.raises(SourceScopeUnavailable):
            capture(legacy, runtime, query, observation)
        assert runtime.budget.sources_used == 0
        assert runtime.cache == scope.bodies == scope.dense_rows == {}
        assert scope.fusion_v2 is None


@pytest.mark.parametrize("changed", [False, True])
def test_precapture_space_replacement_keeps_guarded_public_fallback(legacy, transport, monkeypatch, changed):
    with serving_query(legacy, transport, monkeypatch) as (runtime, scope, observation):
        scope.space = replace(scope.space, **({"space_key": "b" * 64} if changed else {}))
        snapshot = capture(legacy, runtime, "source", observation)
        assert snapshot.source_admission is None
        assert snapshot.baseline_candidates
        assert all(row.doc_id == legacy[0].id for row in snapshot.baseline_candidates)
        assert runtime.budget.sources_used == 1
        if changed:
            assert snapshot.vector_chunk_ids == ()
            assert snapshot.vector_results == () and snapshot.trigram_chunk_ids
            assert set(scope.validate()) == set()


@pytest.mark.parametrize("mutation", ["digest", "compatibility", "space"])
def test_prepared_binding_cannot_change_during_native_metadata_capture(legacy, transport, monkeypatch, mutation):
    from apps.documents.services import graph_candidate_admission as metadata

    with serving_query(legacy, transport, monkeypatch) as (runtime, scope, observation):
        original = metadata.capture_metadata

        def mutate(*args, **kwargs):
            result = original(*args, **kwargs)
            if mutation == "digest":
                scope.query_digest = digest({"version": "unprepared-query-v1", "query": "source"})
            elif mutation == "compatibility":
                scope.query_compatibility = "b" * 64
            else:
                scope.space = replace(scope.space)
            return result

        monkeypatch.setattr(metadata, "capture_metadata", mutate)
        with pytest.raises(SourceScopeUnavailable):
            capture(legacy, runtime, "source", observation)
        assert runtime.budget.sources_used == 0
        assert runtime.cache == scope.bodies == scope.dense_rows == {}
        assert scope.fusion_v2 is None
