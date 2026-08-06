"""Pure response-shape tests for local reranker parsing."""

from types import SimpleNamespace

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


def test_parse_rerank_results_deduplicates_and_skips_invalid_indexes():
    chunks = [SimpleNamespace(pk=101), SimpleNamespace(pk=202)]
    body = {
        "results": [
            {"index": 0},
            {"index": 0},
            {"index": -1},
            {"index": 8},
            {"index": "1"},
            "not-a-result",
        ]
    }

    assert parse_rerank_results(body, chunks) == [101]


def test_score_parsers_accept_supported_response_shapes():
    assert parse_score_results(
        {"data": [{"index": 0, "score": 0.75}, {"index": 1, "relevance_score": 1}]}
    ) == [(0, 0.75), (1, 1.0)]
    assert parse_single_score({"results": [{"relevance_score": 0.25}]}) == 0.25
