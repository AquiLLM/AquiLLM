"""Token-budget contracts for local reranker query/document pairs."""

from __future__ import annotations

from apps.documents.services.chunk_rerank_budget import (
    count_rerank_tokens,
    trim_rerank_pair,
)


def test_trim_rerank_pair_respects_token_budget_and_preserves_normal_query():
    query = "attensity calibration"
    document = "evidence " * 4000

    trimmed_query, trimmed_document = trim_rerank_pair(
        query,
        document,
        max_pair_tokens=900,
        reserve_tokens=64,
    )

    assert trimmed_query == query
    assert count_rerank_tokens(trimmed_query, trimmed_document) <= 836
    assert trimmed_document
    assert len(trimmed_document) < len(document)


def test_trim_rerank_pair_is_deterministic_for_unicode_academic_text():
    first = trim_rerank_pair(
        "β calibration",
        "λ evidence " * 2000,
        max_pair_tokens=900,
        reserve_tokens=64,
    )
    second = trim_rerank_pair(
        "β calibration",
        "λ evidence " * 2000,
        max_pair_tokens=900,
        reserve_tokens=64,
    )

    assert first == second


def test_oversized_query_leaves_room_for_document_evidence():
    trimmed_query, trimmed_document = trim_rerank_pair(
        "query " * 2000,
        "document evidence " * 2000,
        max_pair_tokens=300,
        reserve_tokens=40,
    )

    assert trimmed_query
    assert trimmed_document
    assert count_rerank_tokens(trimmed_query, trimmed_document) <= 260
