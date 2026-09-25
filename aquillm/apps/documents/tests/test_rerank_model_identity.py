"""Unverified server-default models cannot supply comparable cached scores."""

from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import UUID

from apps.documents.services import rag_cache
from apps.documents.services.chunk_rerank import rerank_chunks_scored


def _row(pk, content):
    return SimpleNamespace(
        pk=pk,
        content=content,
        doc_id=UUID("00000000-0000-0000-0000-000000000001"),
        chunk_number=pk,
        modality="text",
        Modality=SimpleNamespace(IMAGE="image"),
    )


def _configure(monkeypatch):
    monkeypatch.setattr(
        "apps.documents.services.chunk_rerank.rerank_provider", lambda: "local"
    )
    monkeypatch.setattr(
        "apps.documents.services.chunk_rerank_local_scored.rerank_model_is_qwen3_vl",
        lambda: False,
    )
    monkeypatch.setattr(
        "apps.documents.services.chunk_rerank_local_scored.rerank_model_revision",
        lambda: "pinned",
    )
    monkeypatch.setattr(rag_cache, "get_cached_rerank_capability", lambda *_: None)


def test_no_model_rerank_payload_keeps_ranking_but_discards_numeric_scores(monkeypatch):
    _configure(monkeypatch)
    payloads = []

    def post(_endpoint, *, json, **_kwargs):
        payloads.append(json)
        response = MagicMock(status_code=400 if "model" in json else 200)
        response.json.return_value = {
            "results": [
                {"index": 1, "relevance_score": 0.9},
                {"index": 0, "relevance_score": 0.1},
            ]
        }
        return response

    monkeypatch.setattr(
        "apps.documents.services.chunk_rerank_local_scored.requests.post", post
    )
    result = rerank_chunks_scored(object, "question", [_row(10, "a"), _row(20, "b")], 1)
    assert any("model" not in payload for payload in payloads)
    assert result.ranked_ids == (20,)
    assert result.score_set.status == "unavailable"
    assert result.score_set.scorer_fingerprint == ""
    assert result.score_set.scores == ()


def test_no_model_batch_score_payload_cannot_claim_configured_model(monkeypatch):
    _configure(monkeypatch)
    payloads = []

    def post(endpoint, *, json, **_kwargs):
        response = MagicMock()
        if endpoint.rsplit("/", 1)[-1] == "rerank":
            response.status_code = 404
        elif "model" in json:
            response.status_code = 400
        else:
            response.status_code = 200
        payloads.append((endpoint, json))
        response.json.return_value = {
            "data": [
                {"index": 0, "score": 0.1},
                {"index": 1, "score": 0.9},
            ]
        }
        return response

    monkeypatch.setattr(
        "apps.documents.services.chunk_rerank_local_scored.requests.post", post
    )
    result = rerank_chunks_scored(object, "question", [_row(10, "a"), _row(20, "b")], 1)
    assert any(
        "score" in endpoint and "model" not in payload for endpoint, payload in payloads
    )
    assert result.ranked_ids == (20,)
    assert result.score_set.status == "unavailable"
    assert result.score_set.scorer_fingerprint == ""
    assert result.score_set.scores == ()
