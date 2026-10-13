"""Actual dense/lexical rank capture keeps provenance and optional policy controls."""

from dataclasses import fields, replace
from types import SimpleNamespace

import pytest
from django.conf import settings

from apps.documents.models import TextChunk
from apps.documents.services.chunk_search_candidates import collect_hybrid_candidate_snapshot
from apps.documents.services.chunk_search_scope import bind_search_scope, freeze_search_scope, reset_search_scope
from apps.documents.services.hybrid_graph_authorization import HybridGraphRetrievalDependencies
from apps.documents.services.source_loading import source_runtime_scope
from apps.documents.tests.lexical_owner_support import unchunked as _legacy_fixture
from apps.documents.tests.test_embedding_generation import stage
from apps.documents.tests.test_embedding_generation_query import runtime_for
from apps.documents.tests.test_lexical_publication import publish
from apps.documents.tests.test_lexical_relevance import replace_chunks
from lib.embeddings.tests.test_document_provider import transport as transport
from lib.knowledge_graph.retrieval_config import load_hybrid_retrieval_settings

legacy = _legacy_fixture

pytestmark = pytest.mark.django_db(transaction=True)


def native_dependencies(authorization):
    from apps.knowledge_graph.retrieval.production_runtime import ProductionHybridBranchRuntime
    from apps.knowledge_graph.projection.identifiers import HmacSha256ProjectionIdentifierCodec

    configured = load_hybrid_retrieval_settings({})
    selected = SimpleNamespace(**{f.name: getattr(configured, f.name) for f in fields(configured)})
    selected.graph_remediation_mode = "v2"
    selected.memgraph_traversal_enabled = selected.graph_direct_enabled = True
    graph = ProductionHybridBranchRuntime(
        authorization=authorization, settings=selected, topology_loader=object(),
        codec=HmacSha256ProjectionIdentifierCodec(b"secret", key_version="key-v1"),
    )
    return HybridGraphRetrievalDependencies(graph, selected, graph.materialize)


def capture(legacy, runtime, query, observation=None, dependencies=None):
    from apps.knowledge_graph.retrieval.expansion import get_graph_expansion_config

    return collect_hybrid_candidate_snapshot(
        TextChunk, query, 10, (legacy[0],), query_embedding=observation,
        graph_config=get_graph_expansion_config(), authorization_context=runtime.authorization,
        app_config_getter=lambda _: SimpleNamespace(vector_top_k=45, trigram_top_k=45),
        _graph_dependencies=dependencies or native_dependencies(runtime.authorization),
    )


@pytest.mark.parametrize("protected", [False, True])
def test_actual_dense_exact_trigram_ranks_and_full_seeds_survive_admission(
    legacy, transport, monkeypatch, protected
):
    from apps.documents.services import embedding_index_store as store
    from apps.documents.services.chunk_search_graph_seeds import build_graph_seeds
    from apps.knowledge_graph.retrieval.expansion import get_graph_expansion_config
    from lib.embeddings.document_provider import observe_query_embedding

    rows = replace_chunks(legacy, ["ATP distractor"] * 31 + ["ATP BRCA1 answer"])
    legacy = (legacy[0], replace(legacy[1], source_hash=legacy[0].full_text_hash))
    _, lease, _, _ = stage(legacy, transport)
    assert store.activate_embedding_generation(lease, using="default") == "activated"
    monkeypatch.setattr(settings, "DOCUMENT_EMBEDDING_SPACE_KEY", transport[0].space_key)
    monkeypatch.setattr(settings, "RAG_RRF_ENABLED", True, raising=False)
    monkeypatch.setattr(settings, "RAG_PROTECTED_EXACT_ENABLED", protected, raising=False)
    monkeypatch.setattr(settings, "RAG_CANDIDATE_MULTIPLIER", 4.0, raising=False)
    monkeypatch.setattr(settings, "RAG_SHORT_QUERY_CANDIDATE_SCALE", 1.0, raising=False)
    runtime, _ = runtime_for(legacy)
    with source_runtime_scope(runtime):
        scope = freeze_search_scope((legacy[0],), using="default", authorization=runtime.authorization)
        token = bind_search_scope(scope, "ATP BRCA1", graph_enabled=True)
        try:
            observation = observe_query_embedding("ATP BRCA1", space=transport[0], budget=runtime.budget)
            snapshot = capture(legacy, runtime, "ATP BRCA1", observation)
            plan = snapshot.source_admission
            channels = dict(plan.channels)
            assert len(snapshot.vector_chunk_ids) == 32
            assert len(snapshot.baseline_candidates) == 25
            assert rows[-1].pk in plan.initial_ids  # exact lane cannot be starved by dense
            assert snapshot.graph_seeds == build_graph_seeds(
                channels["dense"], channels["trigram"], channels["exact"], get_graph_expansion_config()
            )
            assert all(not hasattr(r, "content") for refs in channels.values() for r in refs)
            assert set(scope.dense_rows) <= {r.pk for r in snapshot.baseline_candidates}
            assert bool(snapshot.fusion_v2.candidates[0].protected_exact) is protected
        finally:
            reset_search_scope(token)


@pytest.mark.parametrize("eligible", [False, True])
def test_actual_fts_readiness_and_immutable_binding_capture(legacy, monkeypatch, eligible):
    replace_chunks(legacy, ["assay apparatus", "assay result"])
    if eligible:
        publish(legacy)
    monkeypatch.setattr(settings, "RAG_FULL_TEXT_ENABLED", True, raising=False)
    runtime, _ = runtime_for(legacy)
    with source_runtime_scope(runtime):
        scope = freeze_search_scope((legacy[0],), using="default", authorization=runtime.authorization)
        token = bind_search_scope(scope, "assay", graph_enabled=True)
        try:
            snapshot = capture(legacy, runtime, "assay")
            assert snapshot.full_text_coverage.status == ("ready" if eligible else "incomplete")
            refs = dict(snapshot.source_admission.channels)["full_text"]
            assert bool(refs) is eligible
            if eligible:
                assert all(type(r.fts_binding) is str for r in refs)
                original = refs[0].fts_binding
                scope.fts_rows[refs[0].pk]["extra"] = "mutation"
                assert refs[0].fts_binding == original
        finally:
            reset_search_scope(token)


def test_changed_fts_binding_withdraws_fts_only_body_in_deferred_read(legacy, monkeypatch):
    from apps.documents.services import graph_candidate_admission as metadata

    rows = replace_chunks(legacy, ["assay apparatus"])
    publish(legacy)
    monkeypatch.setattr(settings, "RAG_FULL_TEXT_ENABLED", True, raising=False)
    monkeypatch.setattr(settings, "RAG_TRIGRAM_SIMILARITY_MIN", 1.0, raising=False)
    runtime, _ = runtime_for(legacy)
    with source_runtime_scope(runtime):
        scope = freeze_search_scope((legacy[0],), using="default", authorization=runtime.authorization)
        token = bind_search_scope(scope, "assay", graph_enabled=True)
        original = metadata.capture_metadata

        def mutate(*args, **kwargs):
            result = original(*args, **kwargs)
            binding = TextChunk.objects.get(pk=rows[0].pk).search_vector_binding
            TextChunk.objects.filter(pk=rows[0].pk).update(search_vector_binding={**binding, "extra": "unknown"})
            return result

        monkeypatch.setattr(metadata, "capture_metadata", mutate)
        try:
            snapshot = capture(legacy, runtime, "assay")
            assert snapshot.full_text_chunk_ids == (rows[0].pk,)
            assert snapshot.baseline_candidates == ()
            assert scope.bodies == {} and runtime.cache == {} and scope.fts_rows == {}
            assert rows[0].pk in scope.fts_invalid_ids
        finally:
            reset_search_scope(token)


def test_withdrawn_dense_only_deferred_seed_cannot_influence_final_graph_publication(
    legacy, transport, monkeypatch
):
    from apps.documents.models import EmbeddingGeneration
    from apps.documents.services import embedding_index_store as store
    from apps.documents.services.graph_admission_seeds import native_seed_rows, validate_final_native_seeds
    from apps.documents.services.representation_contracts import SourceScopeUnavailable
    from apps.knowledge_graph.retrieval.graph_seed_allocation_v2 import validated_seed_baseline
    from lib.embeddings.document_provider import observe_query_embedding

    replace_chunks(legacy, ["source fragment"] * 32)
    legacy = (legacy[0], replace(legacy[1], source_hash=legacy[0].full_text_hash))
    _, lease, _, _ = stage(legacy, transport)
    assert store.activate_embedding_generation(lease, using="default") == "activated"
    monkeypatch.setattr(settings, "DOCUMENT_EMBEDDING_SPACE_KEY", transport[0].space_key)
    monkeypatch.setattr(settings, "RAG_TRIGRAM_SIMILARITY_MIN", 1.0, raising=False)
    monkeypatch.setattr(settings, "RAG_CANDIDATE_MULTIPLIER", 4.0, raising=False)
    monkeypatch.setattr(settings, "RAG_SHORT_QUERY_CANDIDATE_SCALE", 1.0, raising=False)
    runtime, _ = runtime_for(legacy)
    with source_runtime_scope(runtime):
        scope = freeze_search_scope((legacy[0],), using="default", authorization=runtime.authorization)
        token = bind_search_scope(scope, "assay", graph_enabled=True)
        try:
            observation = observe_query_embedding("assay", space=transport[0], budget=runtime.budget)
            dependencies = native_dependencies(runtime.authorization)
            dependencies.settings.graph_direct_enabled = False
            dependencies.settings.graph_extended_enabled = True
            snapshot = capture(legacy, runtime, "assay", observation, dependencies)
            assert snapshot.trigram_chunk_ids == snapshot.exact_chunk_ids == ()
            deferred = {s.chunk_id for s in snapshot.graph_seeds} - {r.pk for r in snapshot.baseline_candidates}
            assert deferred and deferred.isdisjoint(scope.dense_rows)
            view, _ = validated_seed_baseline(snapshot, authorization=scope.authorization, settings=dependencies.settings)
            assert view is not None
            before = (runtime.cache.copy(), scope.bodies.copy(), scope.dense_rows.copy(), scope.fusion_v2)
            assert EmbeddingGeneration.objects.filter(pk=lease.generation_key).update(state="failed") == 1
            with pytest.raises(SourceScopeUnavailable):
                native_seed_rows(view)
            # This is also the exact final owner check before staged publication;
            # returned rows alone cannot detect an unreturned seed's withdrawal.
            keys = set(scope.validate())
            with pytest.raises(SourceScopeUnavailable):
                validate_final_native_seeds(snapshot, keys)
            assert (runtime.cache, scope.bodies, scope.dense_rows) == before[:3]
            assert scope.fusion_v2 is before[3] and runtime.budget.sources_used == 25
        finally:
            reset_search_scope(token)
