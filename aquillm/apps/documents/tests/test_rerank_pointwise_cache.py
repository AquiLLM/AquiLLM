"""Rerank HTTP client result and capability caches."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

from django.test import override_settings

from apps.documents.services import rag_cache
from apps.documents.services.chunk_rerank_local_vllm import rerank_via_local_vllm


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
        "apps.documents.services.chunk_rerank_local_vllm.parse_single_score",
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
        "apps.documents.services.chunk_rerank_local_vllm.parse_single_score",
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
