"""Provider score capture and provenance fixtures."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import UUID

from django.test import override_settings

DOC_A = UUID("00000000-0000-0000-0000-000000000001")
DOC_B = UUID("00000000-0000-0000-0000-000000000002")


def _row(pk, content, doc_id=DOC_A):
    return SimpleNamespace(
        pk=pk,
        content=content,
        doc_id=doc_id,
        chunk_number=pk,
        modality="text",
        Modality=SimpleNamespace(IMAGE="image"),
    )


def test_local_single_score_keeps_successful_retry_pair(monkeypatch):
    from apps.documents.services.chunk_rerank_local_scored_http import (
        _score_one_document,
    )

    payloads = []

    def post(_endpoint, *, json, **_kwargs):
        payloads.append(json)
        response = MagicMock(status_code=400 if len(payloads) == 1 else 200)
        response.json.return_value = {"score": 0.8}
        return response

    monkeypatch.setattr(
        "apps.documents.services.chunk_rerank_local_scored.requests.post", post
    )
    result = _score_one_document(
        endpoint="http://test/score",
        index=0,
        query="question " * 300,
        document="evidence " * 300,
        headers={},
        timeout=1,
        model_name="m",
        pair_token_limit=128,
        reserve_tokens=16,
    )
    assert len(payloads) == 2
    assert result[0:2] == (0, 0.8)
    assert result[2] == (payloads[1]["text_1"], payloads[1]["text_2"])
    assert result[2] != (payloads[0]["text_1"], payloads[0]["text_2"])


def test_scored_local_rerank_captures_scores_without_extra_inference(monkeypatch):
    from apps.documents.services import rag_cache
    from apps.documents.services.chunk_rerank import rerank_chunks_scored

    rows = [_row(10, "alpha"), _row(20, "beta", DOC_B)]
    monkeypatch.setattr(
        "apps.documents.services.chunk_rerank.rerank_provider", lambda: "local"
    )
    monkeypatch.setattr(
        "apps.documents.services.chunk_rerank_local_scored.rerank_model_is_qwen3_vl",
        lambda: True,
    )
    monkeypatch.setattr(
        rag_cache,
        "get_cached_rerank_capability",
        lambda *_: {"endpoint": "http://test/score", "shape": "score_single_text_pair"},
    )
    payloads = []

    def post(_endpoint, *, json, **_kwargs):
        payloads.append(json)
        response = MagicMock(status_code=200)
        response.json.return_value = {"score": 0.9 if json["text_2"] == "beta" else 0.7}
        return response

    monkeypatch.setattr(
        "apps.documents.services.chunk_rerank_local_scored.requests.post", post
    )
    result = rerank_chunks_scored(object, "question", rows, 1)
    assert result.ranked_ids == (20,)
    assert [(score.chunk_pk, score.value) for score in result.score_set.scores] == [
        (10, 0.7),
        (20, 0.9),
    ]
    assert result.score_set.status == "complete"
    assert len(payloads) == 2


def test_cohere_scores_stably_break_ties_by_input_order(monkeypatch):
    from apps.documents.services.chunk_rerank import rerank_chunks_scored

    monkeypatch.setattr(
        "apps.documents.services.chunk_rerank.rerank_provider", lambda: "cohere"
    )
    rows = [_row(10, "alpha"), _row(20, "beta", DOC_B)]
    response = SimpleNamespace(
        results=[
            SimpleNamespace(document=SimpleNamespace(id=20), relevance_score=0.7),
            SimpleNamespace(document=SimpleNamespace(id=10), relevance_score=0.7),
        ]
    )
    client = SimpleNamespace(rerank=lambda **_kwargs: response)
    monkeypatch.setattr(
        "apps.documents.services.chunk_rerank.apps.get_app_config",
        lambda _name: SimpleNamespace(cohere_client=client),
    )
    result = rerank_chunks_scored(object, "question", rows, 2)
    assert result.ranked_ids == (10, 20)
    assert [(score.chunk_pk, score.value) for score in result.score_set.scores] == [
        (10, 0.7),
        (20, 0.7),
    ]


def test_scored_cohere_requests_full_pool_when_top_k_is_smaller(monkeypatch):
    from apps.documents.services.chunk_rerank import rerank_chunks_scored

    monkeypatch.setattr(
        "apps.documents.services.chunk_rerank.rerank_provider", lambda: "cohere"
    )
    rows = [_row(10, "alpha"), _row(20, "beta", DOC_B)]
    requested_top_n = []

    def rerank(**kwargs):
        requested_top_n.append(kwargs["top_n"])
        return SimpleNamespace(
            results=[
                SimpleNamespace(document=SimpleNamespace(id=20), relevance_score=0.9),
                SimpleNamespace(document=SimpleNamespace(id=10), relevance_score=0.7),
            ][: kwargs["top_n"]]
        )

    monkeypatch.setattr(
        "apps.documents.services.chunk_rerank.apps.get_app_config",
        lambda _name: SimpleNamespace(cohere_client=SimpleNamespace(rerank=rerank)),
    )
    result = rerank_chunks_scored(object, "question", rows, 1)
    assert requested_top_n == [2]
    assert result.ranked_ids == (20,)
    assert result.score_set.status == "complete"
    assert len(result.score_set.scores) == 2


def test_cohere_without_scores_keeps_top_k_ranking(monkeypatch):
    from apps.documents.services.chunk_rerank import rerank_chunks_scored

    monkeypatch.setattr(
        "apps.documents.services.chunk_rerank.rerank_provider", lambda: "cohere"
    )
    response = SimpleNamespace(
        results=[
            SimpleNamespace(document=SimpleNamespace(id=20)),
            SimpleNamespace(document=SimpleNamespace(id=10)),
        ]
    )
    client = SimpleNamespace(rerank=lambda **_kwargs: response)
    monkeypatch.setattr(
        "apps.documents.services.chunk_rerank.apps.get_app_config",
        lambda _name: SimpleNamespace(cohere_client=client),
    )
    result = rerank_chunks_scored(object, "question", [_row(10, "a"), _row(20, "b")], 1)
    assert result.ranked_ids == (20,)
    assert result.score_set.status == "unavailable"


def test_local_rerank_scores_stably_break_ties_and_missing_scores_are_unavailable(
    monkeypatch,
):
    from apps.documents.services import rag_cache
    from apps.documents.services.chunk_rerank import rerank_chunks_scored

    monkeypatch.setattr(
        "apps.documents.services.chunk_rerank.rerank_provider", lambda: "local"
    )
    monkeypatch.setattr(
        "apps.documents.services.chunk_rerank_local_scored.rerank_model_is_qwen3_vl",
        lambda: False,
    )
    monkeypatch.setattr(rag_cache, "get_cached_rerank_capability", lambda *_: None)
    rows = [_row(10, "alpha"), _row(20, "beta", DOC_B)]
    scored = True

    def post(_endpoint, *, json, **_kwargs):
        response = MagicMock(status_code=200)
        response.json.return_value = {
            "results": [
                {"index": 1, **({"relevance_score": 0.7} if scored else {})},
                {"index": 0, **({"relevance_score": 0.7} if scored else {})},
            ]
        }
        return response

    monkeypatch.setattr(
        "apps.documents.services.chunk_rerank_local_scored.requests.post", post
    )
    result = rerank_chunks_scored(object, "question", rows, 2)
    assert result.ranked_ids == (10, 20)
    assert result.score_set.status == "complete"
    scored = False
    result = rerank_chunks_scored(object, "another question", rows, 2)
    assert result.ranked_ids == (20, 10)
    assert result.score_set.status == "unavailable"
    assert result.score_set.scores == ()


def test_scored_local_rerank_requests_full_pool_when_top_k_is_smaller(monkeypatch):
    from apps.documents.services import rag_cache
    from apps.documents.services.chunk_rerank import rerank_chunks_scored

    monkeypatch.setattr(
        "apps.documents.services.chunk_rerank.rerank_provider", lambda: "local"
    )
    monkeypatch.setattr(
        "apps.documents.services.chunk_rerank_local_scored.rerank_model_is_qwen3_vl",
        lambda: False,
    )
    monkeypatch.setattr(rag_cache, "get_cached_rerank_capability", lambda *_: None)
    top_ns = []

    def post(_endpoint, *, json, **_kwargs):
        top_ns.append(json["top_n"])
        response = MagicMock(status_code=200)
        response.json.return_value = {
            "results": [
                {"index": 1, "relevance_score": 0.9},
                {"index": 0, "relevance_score": 0.7},
            ][: json["top_n"]]
        }
        return response

    monkeypatch.setattr(
        "apps.documents.services.chunk_rerank_local_scored.requests.post", post
    )
    result = rerank_chunks_scored(
        object, "question", [_row(10, "alpha"), _row(20, "beta")], 1
    )
    assert top_ns == [2]
    assert result.ranked_ids == (20,)
    assert result.score_set.status == "complete"
    assert len(result.score_set.scores) == 2


@override_settings(RAG_CACHE_ENABLED=True)
def test_scored_cache_rebinds_when_capability_endpoint_changes(monkeypatch):
    from apps.documents.services import rag_cache
    from apps.documents.services.chunk_rerank import rerank_chunks_scored

    monkeypatch.setattr(
        "apps.documents.services.chunk_rerank.rerank_provider", lambda: "local"
    )
    monkeypatch.setattr(
        "apps.documents.services.chunk_rerank_local_scored.rerank_model_is_qwen3_vl",
        lambda: True,
    )
    monkeypatch.setattr(
        "apps.documents.services.chunk_rerank_local_scored.rerank_model_revision",
        lambda: "pinned-revision",
    )
    endpoint = ["http://test/score-a"]
    monkeypatch.setattr(
        rag_cache,
        "get_cached_rerank_capability",
        lambda *_: {"endpoint": endpoint[0], "shape": "score_single_text_pair"},
    )
    posts = []

    def post(url, *, json, **_kwargs):
        posts.append(url)
        response = MagicMock(status_code=200)
        response.json.return_value = {"score": 0.8}
        return response

    monkeypatch.setattr(
        "apps.documents.services.chunk_rerank_local_scored.requests.post", post
    )
    rows = [_row(101, "one")]
    first = rerank_chunks_scored(object, "endpoint-sensitive", rows, 1)
    endpoint[0] = "http://test/score-b"
    second = rerank_chunks_scored(object, "endpoint-sensitive", rows, 1)
    assert posts == ["http://test/score-a", "http://test/score-b"]
    assert first.score_set.scorer_fingerprint != second.score_set.scorer_fingerprint
