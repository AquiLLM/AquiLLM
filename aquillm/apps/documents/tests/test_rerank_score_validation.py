"""Score validation and deterministic ranking contracts."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from apps.documents.services import rag_cache
from apps.documents.services.chunk_rerank_local_vllm import rerank_via_local_vllm
from apps.documents.services.chunk_rerank_parse import (
    parse_score_results,
    parse_single_score,
)
from apps.documents.services.chunk_rerank_score import rank_complete_scores


def test_score_parsers_reject_json_booleans():
    assert parse_score_results({"data": [{"index": 0, "score": True}]}) == []
    with pytest.raises(ValueError):
        parse_single_score({"score": True})


def test_equal_scores_keep_original_candidate_order_after_shuffled_completion():
    chunks = [MagicMock(pk=10), MagicMock(pk=20), MagicMock(pk=30)]

    ranked = rank_complete_scores([(2, 0.0), (1, 0.0), (0, 0.0)], chunks, 2)

    assert ranked == [10, 20]


def test_boolean_single_scores_do_not_populate_result_cache(monkeypatch):
    module = "apps.documents.services.chunk_rerank_local_vllm"
    monkeypatch.setattr(f"{module}.rerank_base_url", lambda: "http://reranker/v1")
    monkeypatch.setattr(f"{module}.rerank_model", lambda: "reranker")
    monkeypatch.setattr(f"{module}.rerank_model_is_qwen3_vl", lambda: True)
    monkeypatch.setattr(
        f"{module}.rerank_document_payload", lambda chunk: chunk.content
    )
    monkeypatch.setattr(rag_cache, "get_cached_rerank_result", lambda *_args: None)
    monkeypatch.setattr(
        rag_cache,
        "get_cached_rerank_capability",
        lambda *_args: {
            "endpoint": "http://reranker/score",
            "shape": "score_single_text_pair",
        },
    )
    monkeypatch.setattr(
        rag_cache, "delete_cached_rerank_capability", lambda *_args: None
    )
    stored_rankings: list[list[int]] = []
    monkeypatch.setattr(
        rag_cache,
        "set_cached_rerank_result",
        lambda _query, _candidates, _top_k, _model, ranked: stored_rankings.append(
            ranked
        ),
    )

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {"score": True}

    monkeypatch.setattr(
        "apps.documents.services.chunk_rerank_score.requests.post",
        lambda *_args, **_kwargs: Response(),
    )
    rows = [MagicMock(pk=1, content="a"), MagicMock(pk=2, content="b")]

    class Model:
        objects = MagicMock()

    rerank_via_local_vllm(Model, "query", rows, 2)

    assert stored_rankings == []
