"""Rerank HTTP client result and capability caches."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

from django.test import override_settings

from apps.documents.services import rag_cache
from apps.documents.services.chunk_rerank_local_vllm import rerank_via_local_vllm


def test_window_retry_charges_both_transports_and_disallows_silent_truncation():
    from time import monotonic

    from apps.documents.services.chunk_rerank_selection_provider import (
        LocalSelectionScorer,
    )
    from lib.retrieval import TurnBudget, TurnLimits

    scorer = LocalSelectionScorer(
        endpoint="http://test/score",
        shape="score_single_text_pair",
        model_name="m",
        revision="r",
        char_limit=2000,
        pair_limit=1024,
        reserve=256,
        timeout=3,
        deadline=monotonic() + 3,
    )
    budget = TurnBudget(TurnLimits())
    failed, good = MagicMock(status_code=400), MagicMock(status_code=200)
    good.json.return_value = {"score": 0.7}
    with patch(
        "apps.documents.services.chunk_rerank_selection_provider.requests.post",
        side_effect=[failed, good],
    ) as post:
        scored = scorer.score_budgeted_pair(
            ("whole question", "abcd" * 100), 2, budget=budget, phase="acquisition"
        )
    assert scored == (0.7, ("whole question", "abcd" * 50))
    assert budget.pairs_used["acquisition"] == 2
    assert all(
        "truncate_prompt_tokens" not in c.kwargs["json"] for c in post.call_args_list
    )
    assert all(0 < c.kwargs["timeout"] <= 2 for c in post.call_args_list)


@override_settings(RAG_CACHE_ENABLED=True)
def test_window_cache_requires_revision_and_rejects_late_writes():
    from time import monotonic
    from types import SimpleNamespace
    from uuid import UUID

    from apps.documents.services.chunk_rerank_window_adapter import (
        WindowSelectionScorer,
    )
    from lib.retrieval import TurnBudget, TurnLimits

    class Provider:
        calls = 0

        def score_pair(self, pair, timeout_seconds):
            self.calls += 1
            return (0.5, pair)

    provider = Provider()

    def counter(q, d):
        return len(q) + len(d)

    row = SimpleNamespace(
        pk=989,
        doc_id=UUID(int=989),
        chunk_number=0,
        content="cache evidence",
        source_revision="one",
    )

    def adapter(budget):
        return WindowSelectionScorer(
            provider,
            budget=budget,
            pair_counter=counter,
            scorer_identity="cache-fixture-unique",
            deadline=monotonic() + 2,
            cache_enabled=True,
        )

    first = adapter(TurnBudget(TurnLimits())).score_windows("q-cache", (row,))
    second = adapter(TurnBudget(TurnLimits())).score_windows("q-cache", (row,))
    assert first == second
    assert provider.calls == 1
    row.source_revision = "two"
    changed = adapter(TurnBudget(TurnLimits())).score_windows("q-cache", (row,))
    assert (
        changed.scores[0].effective_pair_fingerprint
        != first.scores[0].effective_pair_fingerprint
    )
    assert provider.calls == 2
    closed = TurnBudget(TurnLimits())
    closed.close("cancelled")
    assert adapter(closed).score_windows("q-cache", (row,)).status == "unavailable"
    assert provider.calls == 2


def test_verified_window_rejects_server_usage_that_hides_truncation():
    from time import monotonic

    from apps.documents.services.chunk_rerank_selection_provider import (
        LocalSelectionScorer,
    )
    from lib.retrieval import TurnBudget, TurnLimits

    scorer = LocalSelectionScorer(
        endpoint="http://test/score",
        shape="score_single_text_pair",
        model_name="m",
        revision="r",
        char_limit=2000,
        pair_limit=1024,
        reserve=256,
        timeout=3,
        deadline=monotonic() + 3,
    )
    scorer.verified_pair_counter = lambda q, d: 100
    response = MagicMock(status_code=200)
    response.json.return_value = {"score": 0.8, "usage": {"prompt_tokens": 50}}
    budget = TurnBudget(TurnLimits())
    with patch(
        "apps.documents.services.chunk_rerank_selection_provider.requests.post",
        return_value=response,
    ):
        assert (
            scorer.score_budgeted_pair(("q", "source"), 1, budget=budget, phase="final")
            is None
        )
    assert budget.text_used["tokenized"] == 7


@override_settings(RAG_CACHE_ENABLED=True)
@patch("apps.documents.services.chunk_rerank_local_vllm.requests.post")
@patch("apps.documents.services.chunk_rerank_local_vllm.rerank_model_is_qwen3_vl")
@patch("apps.documents.services.chunk_rerank_local_vllm.rerank_model")
@patch("apps.documents.services.chunk_rerank_local_vllm.rerank_base_url")
@patch("apps.documents.services.chunk_rerank_local_vllm.parse_rerank_results")
def test_rerank_result_cache_second_call_skips_http(
    mock_parse,
    mock_base_url,
    mock_model,
    mock_vl,
    mock_post,
):
    mock_vl.return_value = False
    mock_base_url.return_value = "http://test/v1"
    mock_model.return_value = "m1"
    mock_parse.return_value = [7, 3, 1]

    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"ok": True}
    mock_post.return_value = resp

    chunks = [MagicMock(pk=i, content=f"t{i}") for i in (1, 2, 3, 4, 5, 6, 7)]

    class MC:
        objects = MagicMock()

    ordered = MagicMock()
    with patch(
        "apps.documents.services.chunk_rerank_local_vllm.ordered_queryset_from_ids",
        return_value=ordered,
    ):
        rerank_via_local_vllm(MC, "same query", chunks, 3)
        posts_after_first = mock_post.call_count
        rerank_via_local_vllm(MC, "same query", chunks, 3)
    assert mock_post.call_count == posts_after_first


@override_settings(RAG_CACHE_ENABLED=True)
@patch("apps.documents.services.chunk_rerank_local_vllm.requests.post")
@patch("apps.documents.services.chunk_rerank_local_vllm.rerank_document_payload")
@patch("apps.documents.services.chunk_rerank_local_vllm.rerank_model")
@patch("apps.documents.services.chunk_rerank_local_vllm.rerank_base_url")
def test_rerank_cache_hit_skips_multimodal_payload_work(
    mock_base_url,
    mock_model,
    mock_payload,
    mock_post,
):
    mock_base_url.return_value = "http://test/v1"
    mock_model.return_value = "m1"

    chunks = [MagicMock(pk=i, content=f"t{i}") for i in (1, 2, 3)]

    class MC:
        objects = MagicMock()

    ordered = MagicMock()
    with patch(
        "apps.documents.services.chunk_rerank_local_vllm.rag_cache.get_cached_rerank_result",
        return_value=[3, 1, 2],
    ):
        with patch(
            "apps.documents.services.chunk_rerank_local_vllm.ordered_queryset_from_ids",
            return_value=ordered,
        ):
            out = rerank_via_local_vllm(MC, "cached query", chunks, 3)
    mock_payload.assert_not_called()
    mock_post.assert_not_called()
    assert out is ordered


@override_settings(RAG_CACHE_ENABLED=True)
@patch("apps.documents.services.chunk_rerank_local_vllm.logger")
@patch("apps.documents.services.chunk_rerank_local_vllm.requests.post")
@patch("apps.documents.services.chunk_rerank_local_vllm.rerank_model")
@patch("apps.documents.services.chunk_rerank_local_vllm.rerank_base_url")
def test_rerank_logs_cache_hit_without_query_text(
    mock_base_url,
    mock_model,
    mock_post,
    mock_logger,
):
    mock_base_url.return_value = "http://test/v1"
    mock_model.return_value = "m1"
    chunks = [MagicMock(pk=i, content="secret-query-text") for i in (1, 2, 3)]

    class MC:
        objects = MagicMock()

    with patch(
        "apps.documents.services.chunk_rerank_local_vllm.rag_cache.get_cached_rerank_result",
        return_value=[3, 2, 1],
    ):
        with patch(
            "apps.documents.services.chunk_rerank_local_vllm.ordered_queryset_from_ids",
            return_value=MagicMock(),
        ):
            rerank_via_local_vllm(
                MC, "user query must not appear in metrics log", chunks, 2
            )

    mock_post.assert_not_called()
    hit_calls = [
        c
        for c in mock_logger.info.call_args_list
        if c.args and c.args[0] == "obs.rag.rerank_cache_hit"
    ]
    assert hit_calls
    assert hit_calls[0].kwargs == {
        "reason": "completed",
        "count": 0,
        "elapsed_ms": 0.0,
    }
    joined = " ".join(f"{call.args!r} {call.kwargs!r}" for call in hit_calls)
    assert "secret-query-text" not in joined
    assert "user query must not appear" not in joined


@override_settings(RAG_CACHE_ENABLED=True)
def test_rerank_capability_cache_round_trips_endpoint_and_payload_shape():
    capability = {
        "endpoint": "http://reranker/score",
        "shape": "score_single_text_pair",
    }

    rag_cache.set_cached_rerank_capability("http://reranker/v1", "m1", capability)

    assert (
        rag_cache.get_cached_rerank_capability("http://reranker/v1", "m1") == capability
    )


@patch("apps.documents.services.chunk_rerank_local_vllm.ordered_queryset_from_ids")
@patch("apps.documents.services.chunk_rerank_local_vllm.rerank_model_is_qwen3_vl")
@patch("apps.documents.services.chunk_rerank_local_vllm.rerank_model")
@patch("apps.documents.services.chunk_rerank_local_vllm.rerank_base_url")
def test_cached_single_score_capability_skips_batch_and_scores_concurrently(
    mock_base_url,
    mock_model,
    mock_vl,
    mock_ordered,
    monkeypatch,
):
    mock_base_url.return_value = "http://reranker/v1"
    mock_model.return_value = "Qwen/Qwen3-VL-Reranker-2B"
    mock_vl.return_value = True
    rows = [
        MagicMock(pk=index, content=f"document evidence {index}") for index in range(12)
    ]
    mock_ordered.side_effect = lambda _model, identifiers: tuple(identifiers)
    monkeypatch.setattr(
        rag_cache,
        "get_cached_rerank_result",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        rag_cache,
        "get_cached_rerank_capability",
        lambda *_args: {
            "endpoint": "http://reranker/score",
            "shape": "score_single_text_pair",
        },
    )
    monkeypatch.setattr(rag_cache, "set_cached_rerank_result", lambda *_args: None)
    monkeypatch.setattr(rag_cache, "set_cached_rerank_capability", lambda *_args: None)

    lock = threading.Lock()
    active = 0
    peak = 0
    payloads: list[dict] = []

    class Response:
        status_code = 200

        def __init__(self, score):
            self._score = score

        def json(self):
            return {"score": self._score}

    def post(_endpoint, *, json, **_kwargs):
        nonlocal active, peak
        assert not isinstance(json.get("text_2"), list)
        with lock:
            payloads.append(json)
            active += 1
            peak = max(peak, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return Response(float(json["text_2"].rsplit(" ", 1)[-1]))

    monkeypatch.setattr(
        "apps.documents.services.chunk_rerank_local_vllm.requests.post",
        post,
    )
    monkeypatch.setattr(
        "apps.documents.services.chunk_rerank_score.parse_single_score",
        lambda payload: payload["score"],
    )

    ranked = rerank_via_local_vllm(object, "attensity", rows, 4)

    assert ranked == (11, 10, 9, 8)
    assert len(payloads) == len(rows)
    assert 1 < peak <= 6


@patch("apps.documents.services.chunk_rerank_local_vllm.ordered_queryset_from_ids")
@patch("apps.documents.services.chunk_rerank_local_vllm.rerank_model_is_qwen3_vl")
@patch("apps.documents.services.chunk_rerank_local_vllm.rerank_model")
@patch("apps.documents.services.chunk_rerank_local_vllm.rerank_base_url")
def test_cold_qwen_vl_path_avoids_known_invalid_batch_payload(
    mock_base_url,
    mock_model,
    mock_vl,
    mock_ordered,
    monkeypatch,
):
    mock_base_url.return_value = "http://reranker/v1"
    mock_model.return_value = "Qwen/Qwen3-VL-Reranker-2B"
    mock_vl.return_value = True
    rows = [MagicMock(pk=index, content=f"evidence {index}") for index in range(4)]
    mock_ordered.side_effect = lambda _model, identifiers: tuple(identifiers)
    monkeypatch.setattr(rag_cache, "get_cached_rerank_result", lambda *_args: None)
    monkeypatch.setattr(rag_cache, "get_cached_rerank_capability", lambda *_args: None)
    monkeypatch.setattr(rag_cache, "set_cached_rerank_result", lambda *_args: None)
    monkeypatch.setattr(rag_cache, "set_cached_rerank_capability", lambda *_args: None)

    class Response:
        status_code = 200

        def __init__(self, score):
            self._score = score

        def json(self):
            return {"score": self._score}

    payloads: list[dict] = []

    def post(_endpoint, *, json, **_kwargs):
        payloads.append(json)
        assert not isinstance(json.get("text_2"), list)
        return Response(float(json["text_2"].rsplit(" ", 1)[-1]))

    monkeypatch.setattr(
        "apps.documents.services.chunk_rerank_local_vllm.requests.post",
        post,
    )
    monkeypatch.setattr(
        "apps.documents.services.chunk_rerank_score.parse_single_score",
        lambda payload: payload["score"],
    )

    ranked = rerank_via_local_vllm(object, "attensity", rows, 2)

    assert ranked == (3, 2)
    assert len(payloads) == len(rows)


@patch("apps.documents.services.chunk_rerank_local_vllm.rerank_model_is_qwen3_vl")
@patch("apps.documents.services.chunk_rerank_local_vllm.rerank_model")
@patch("apps.documents.services.chunk_rerank_local_vllm.rerank_base_url")
def test_incomplete_or_nonfinite_single_scores_do_not_populate_result_cache(
    mock_base_url,
    mock_model,
    mock_vl,
    monkeypatch,
):
    mock_base_url.return_value = "http://reranker/v1"
    mock_model.return_value = "Qwen/Qwen3-VL-Reranker-2B"
    mock_vl.return_value = True
    rows = [MagicMock(pk=index, content=f"evidence {index}") for index in range(2)]

    class Model:
        objects = MagicMock()

    stored_rankings: list[list[int]] = []
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
        rag_cache,
        "set_cached_rerank_result",
        lambda _query, _candidates, _top_k, _model, ranked: stored_rankings.append(
            ranked
        ),
    )
    monkeypatch.setattr(
        "apps.documents.services.chunk_rerank_local_vllm._score_documents_concurrently",
        lambda **_kwargs: [(0, 0.9), (1, float("nan"))],
    )

    rerank_via_local_vllm(Model, "attensity", rows, 2)

    assert stored_rankings == []
