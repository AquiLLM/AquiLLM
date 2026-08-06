"""Pure response-shape tests for local reranker parsing."""

import math
from types import SimpleNamespace

import pytest

from apps.documents.services.chunk_rerank_parse import (
    parse_rerank_results,
    parse_score_results,
    parse_single_score,
)


def test_parse_rerank_results_accepts_results_and_data_shapes():
    chunks = [SimpleNamespace(pk=101), SimpleNamespace(pk=202)]

    assert parse_rerank_results({"results": [{"index": 1}, {"index": 0}]}, chunks) == [
        202,
        101,
    ]
    assert parse_rerank_results({"data": [{"index": 0}]}, chunks) == [101]
    for malformed in (None, [], {}, {"results": None}, {"data": []}):
        assert parse_rerank_results(malformed, chunks) == []


def test_parse_rerank_results_deduplicates_and_skips_invalid_indexes():
    chunks = [SimpleNamespace(pk=101), SimpleNamespace(pk=202)]
    body = {
        "results": [
            {"index": 0},
            {"index": 0},
            {"index": -1},
            {"index": 8},
            {"index": "1"},
            {"index": True},
            {"index": False},
            {"index": float("nan")},
            {"index": float("inf")},
            {"index": float("-inf")},
            {"index": 0.5},
            {"index": 1.0},
            "not-a-result",
        ]
    }

    assert parse_rerank_results(body, chunks) == [101]


def test_score_parsers_accept_supported_response_shapes():
    assert parse_score_results(
        {"data": [{"index": 0, "score": 0.75}, {"index": 1, "relevance_score": 1}]}
    ) == [(0, 0.75), (1, 1.0)]
    assert parse_single_score({"results": [{"relevance_score": 0.25}]}) == 0.25

    malformed_items = [
        {"index": True, "score": 0.5},
        {"index": 0.5, "score": 0.5},
        {"index": 2, "score": True},
        {"index": 3, "score": float("nan")},
        {"index": 4, "score": float("inf")},
        {"index": 5, "relevance_score": float("-inf")},
    ]
    assert parse_score_results({"results": malformed_items}) == []
    for malformed in (None, [], {}, {"results": None}, {"data": []}):
        assert parse_score_results(malformed) == []

    invalid_single_values = [True, False, float("nan"), float("inf"), float("-inf")]
    invalid_single_shapes = [
        None,
        [],
        {},
        {"score": True},
        {"score": float("nan")},
        {"data": []},
        {"data": [{"score": float("inf")}]},
        {"results": []},
        {"results": [{"relevance_score": float("-inf")}]},
    ]
    for value in [*invalid_single_values, *invalid_single_shapes]:
        with pytest.raises(ValueError, match="Unable to parse score response"):
            parse_single_score(value)

    assert math.isfinite(parse_single_score(0.0))
