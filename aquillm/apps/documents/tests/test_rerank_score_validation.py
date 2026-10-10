"""Score validation and deterministic ranking contracts."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from apps.documents.services import rag_cache
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
    from apps.documents.services import chunk_rerank_local_scored as local
    from apps.documents.tests.test_rerank_score_providers import _configure_verified

    _configure_verified(monkeypatch)
    monkeypatch.setattr(local, "rerank_model_revision", lambda: "synthetic-revision")
    monkeypatch.setattr(rag_cache, "cache_get", lambda *a, **k: None)
    stored_rankings = []
    monkeypatch.setattr(rag_cache, "get_cached_rerank_result", lambda *a, **k: None)
    monkeypatch.setattr(
        rag_cache,
        "set_cached_rerank_result",
        lambda *a, **k: stored_rankings.append((a, k)),
    )
    monkeypatch.setattr(
        rag_cache, "cache_set", lambda *a, **k: stored_rankings.append((a, k))
    )
    calls = []

    class Response:
        status_code = 200
        url = "http://test/score"

        @staticmethod
        def json():
            return {"score": True}

    monkeypatch.setattr(
        "apps.documents.services.chunk_rerank_score.requests.post",
        lambda *_args, **_kwargs: calls.append(_kwargs) or Response(),
    )
    rows = [MagicMock(pk=1, content="a"), MagicMock(pk=2, content="b")]

    class Model:
        objects = MagicMock()

    result = local.rerank_via_local_vllm_scored(Model, "query", rows, 2)

    assert 1 <= len(calls) <= 2
    assert result.score_set.status == "unavailable"
    assert result.ranked_ids == (1, 2)
    assert stored_rankings == []
