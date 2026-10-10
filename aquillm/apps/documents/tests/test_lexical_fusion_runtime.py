"""Selected fusion order survives reranker failure and final source revocation."""

from dataclasses import replace

import pytest
from django.test import override_settings

from apps.documents.models import TextChunk
from apps.documents.services import chunk_rerank, chunk_search
from apps.documents.services.chunk_search_candidates import (
    collect_hybrid_candidate_snapshot,
)
from apps.documents.services.source_loading import source_runtime_scope
from apps.documents.tests.test_embedding_generation import legacy as legacy
from apps.documents.tests.test_embedding_generation import stage
from apps.documents.tests.test_embedding_generation_query import runtime_for
from apps.documents.tests.test_lexical_publication import publish
from apps.documents.tests.test_lexical_relevance import replace_chunks
from lib.embeddings.tests.test_document_provider import transport as transport

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.mark.parametrize("capture", [False, True])
@override_settings(
    RAG_RRF_ENABLED=True,
    RAG_PROTECTED_EXACT_ENABLED=True,
    RAG_CANDIDATE_MULTIPLIER=1,
    RAG_TRIGRAM_SIMILARITY_MIN=1,
)
def test_protected_answer_survives_actual_dense_collection_and_score_failure(
    legacy, transport, monkeypatch, capture
):
    from apps.documents.services import embedding_index_store as store

    rows = replace_chunks(legacy, ["ATP distractor"] * 12 + ["ATP BRCA1 answer"])
    legacy = (legacy[0], replace(legacy[1], source_hash=legacy[0].full_text_hash))
    _, lease, _, _ = stage(legacy, transport)
    assert store.activate_embedding_generation(lease, using="default") == "activated"
    runtime, _ = runtime_for(legacy)
    monkeypatch.setenv(
        "RAG_EVIDENCE_SELECTION_MODE", "adaptive" if capture else "legacy"
    )

    def unavailable(*args, **kwargs):
        raise TimeoutError("synthetic scorer unavailable")

    monkeypatch.setattr(chunk_rerank, "rerank_via_local_vllm", unavailable)
    monkeypatch.setattr(chunk_rerank, "rerank_via_local_vllm_scored", unavailable)
    with (
        override_settings(DOCUMENT_EMBEDDING_SPACE_KEY=transport[0].space_key),
        source_runtime_scope(runtime),
    ):
        vectors, _, ranked, details = chunk_search.text_chunk_search(
            TextChunk,
            "ATP BRCA1",
            10,
            [legacy[0]],
            authorization_context=runtime.authorization,
        )
    assert rows[-1].pk not in [row.pk for row in vectors]
    assert ranked[0].pk == rows[-1].pk
    assert len(ranked) == 10
    if capture:
        assert details["_score_set"]["status"] == "unavailable"
        assert details["_score_set"]["scores"] == []
        assert set(details["_score_set"]["candidate_order"]) == {
            row.pk for row in ranked
        }


@override_settings(RAG_FULL_TEXT_ENABLED=True, RAG_TRIGRAM_SIMILARITY_MIN=1)
def test_title_change_at_final_rerank_rejects_fts_only_row_and_cache(
    legacy, monkeypatch
):
    from apps.documents.services.chunk_search_scope import current_scope

    publish(legacy)
    original = chunk_search._fallback_rerank
    keys = []

    def change_title(model, rows, count):
        keys.append(current_scope().cache_key())
        type(legacy[0]).objects.filter(pk=legacy[0].pk).update(title="changed")
        assert current_scope().cache_key() is None
        return original(model, rows, count)

    runtime, _ = runtime_for(legacy)
    monkeypatch.setattr(chunk_search, "_fallback_rerank", change_title)
    with source_runtime_scope(runtime):
        result = chunk_search.text_chunk_search(
            TextChunk,
            "Synthetic",
            10,
            [legacy[0]],
            authorization_context=runtime.authorization,
        )
    assert keys[0] is not None
    assert result[2] == []


@override_settings(RAG_FULL_TEXT_ENABLED=True)
def test_full_text_shares_existing_trigram_candidate_allowance(legacy):
    replace_chunks(legacy, ["measurement source"] * 20)
    publish(legacy)
    snapshot = collect_hybrid_candidate_snapshot(
        TextChunk, "measurement", 10, [legacy[0]], query_embedding=None
    )
    assert snapshot.full_text_coverage.status == "ready"
    assert (
        len(snapshot.full_text_chunk_ids) + len(snapshot.trigram_chunk_ids)
        <= snapshot.effective_limits.trigram
    )


@override_settings(RAG_FULL_TEXT_ENABLED=True)
def test_incomplete_full_text_keeps_independent_exact_evidence(legacy):
    rows = replace_chunks(legacy, ["ATP BRCA1 source"])
    snapshot = collect_hybrid_candidate_snapshot(
        TextChunk, "ATP BRCA1", 10, [legacy[0]], query_embedding=None
    )
    assert snapshot.full_text_coverage.status == "incomplete"
    assert snapshot.exact_chunk_ids == (rows[0].pk,)


def test_optional_controls_default_off(legacy):
    result = collect_hybrid_candidate_snapshot(
        TextChunk, "source", 10, [legacy[0]], query_embedding=None
    )
    assert result.full_text_coverage.status == "disabled"
    assert result.fusion_v2 is None


@override_settings(RAG_RRF_ENABLED=True)
def test_direct_collector_fusion_is_bound_to_each_query_in_original_runtime(legacy):
    runtime, _ = runtime_for(legacy)
    with source_runtime_scope(runtime):
        first = collect_hybrid_candidate_snapshot(
            TextChunk,
            "ATP",
            10,
            [legacy[0]],
            query_embedding=None,
            authorization_context=runtime.authorization,
        )
        original = runtime._frozen_retrieval_scope
        second = collect_hybrid_candidate_snapshot(
            TextChunk,
            "BRCA1",
            10,
            [legacy[0]],
            query_embedding=None,
            authorization_context=runtime.authorization,
        )
    assert runtime._frozen_retrieval_scope is original
    assert first.fusion_v2.scope_key != second.fusion_v2.scope_key
