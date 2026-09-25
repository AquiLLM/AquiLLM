"""Request routing contracts for local HTTP reranking."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from apps.documents.services import rag_cache
from apps.documents.services.chunk_rerank_budget import count_rerank_tokens
from apps.documents.services.chunk_rerank_local_vllm import rerank_via_local_vllm


class _Response:
    def __init__(self, status_code: int, body: dict | None = None):
        self.status_code = status_code
        self._body = body or {}

    def json(self):
        return self._body


def _configure_reranker(monkeypatch, *, capability=None):
    module = "apps.documents.services.chunk_rerank_local_vllm"
    monkeypatch.setattr(f"{module}.rerank_base_url", lambda: "http://reranker/v1")
    monkeypatch.setattr(f"{module}.rerank_model", lambda: "reranker")
    monkeypatch.setattr(f"{module}.rerank_doc_char_limit", lambda: 2000)
    monkeypatch.setattr(f"{module}.rerank_pair_token_limit", lambda: 1024)
    monkeypatch.setattr(f"{module}.rerank_template_reserve_tokens", lambda: 96)
    monkeypatch.setattr(
        f"{module}.rerank_document_payload", lambda chunk: chunk.content
    )
    monkeypatch.setattr(rag_cache, "get_cached_rerank_result", lambda *_args: None)
    monkeypatch.setattr(
        rag_cache, "get_cached_rerank_capability", lambda *_args: capability
    )
    monkeypatch.setattr(rag_cache, "set_cached_rerank_result", lambda *_args: None)
    monkeypatch.setattr(rag_cache, "set_cached_rerank_capability", lambda *_args: None)


@patch("apps.documents.services.chunk_rerank_local_vllm.ordered_queryset_from_ids")
def test_shared_query_keeps_every_uneven_document_pair_within_budget(
    mock_ordered, monkeypatch
):
    _configure_reranker(monkeypatch)
    monkeypatch.setattr(
        "apps.documents.services.chunk_rerank_local_vllm.rerank_model_is_qwen3_vl",
        lambda: False,
    )
    mock_ordered.side_effect = lambda _model, identifiers: tuple(identifiers)
    rows = [
        MagicMock(pk=1, content="x"),
        MagicMock(pk=2, content="z " * 450),
    ]
    payloads: list[dict] = []

    def post(_endpoint, *, json, **_kwargs):
        payloads.append(json)
        return _Response(200, {"results": [{"index": 0}]})

    monkeypatch.setattr(
        "apps.documents.services.chunk_rerank_local_vllm.requests.post", post
    )

    ranked = rerank_via_local_vllm(object, "query " * 2000, rows, 1)

    assert ranked == (1,)
    first_payload = payloads[0]
    assert all(
        count_rerank_tokens(first_payload["query"], document) <= 928
        for document in first_payload["documents"]
    )


@patch("apps.documents.services.chunk_rerank_local_vllm.ordered_queryset_from_ids")
def test_cached_batch_score_capability_goes_directly_to_score(
    mock_ordered, monkeypatch
):
    capability = {
        "endpoint": "http://reranker/score",
        "shape": "score_batch_text_pairs",
    }
    _configure_reranker(monkeypatch, capability=capability)
    monkeypatch.setattr(
        "apps.documents.services.chunk_rerank_local_vllm.rerank_model_is_qwen3_vl",
        lambda: False,
    )
    mock_ordered.side_effect = lambda _model, identifiers: tuple(identifiers)
    rows = [MagicMock(pk=1, content="a"), MagicMock(pk=2, content="b")]
    endpoints: list[str] = []

    def post(endpoint, **_kwargs):
        endpoints.append(endpoint)
        if endpoint == capability["endpoint"]:
            return _Response(
                200,
                {"data": [{"index": 0, "score": 0.2}, {"index": 1, "score": 0.9}]},
            )
        return _Response(404)

    monkeypatch.setattr(
        "apps.documents.services.chunk_rerank_local_vllm.requests.post", post
    )

    ranked = rerank_via_local_vllm(object, "query", rows, 2)

    assert ranked == (2, 1)
    assert endpoints == ["http://reranker/score"]


@patch("apps.documents.services.chunk_rerank_local_vllm.ordered_queryset_from_ids")
def test_failed_cached_batch_score_falls_back_to_rerank_discovery(
    mock_ordered, monkeypatch
):
    capability = {
        "endpoint": "http://cached/score",
        "shape": "score_batch_text_pairs",
    }
    _configure_reranker(monkeypatch, capability=capability)
    monkeypatch.setattr(
        "apps.documents.services.chunk_rerank_local_vllm.rerank_model_is_qwen3_vl",
        lambda: False,
    )
    deleted: list[tuple[str, str]] = []
    monkeypatch.setattr(
        rag_cache,
        "delete_cached_rerank_capability",
        lambda base, model: deleted.append((base, model)),
    )
    mock_ordered.side_effect = lambda _model, identifiers: tuple(identifiers)
    rows = [MagicMock(pk=1, content="a"), MagicMock(pk=2, content="b")]
    endpoints: list[str] = []

    def post(endpoint, **_kwargs):
        endpoints.append(endpoint)
        if endpoint == capability["endpoint"]:
            return _Response(500)
        return _Response(200, {"results": [{"index": 1}]})

    monkeypatch.setattr(
        "apps.documents.services.chunk_rerank_local_vllm.requests.post", post
    )

    ranked = rerank_via_local_vllm(object, "query", rows, 1)

    assert ranked == (2,)
    assert endpoints[0] == "http://cached/score"
    assert any(endpoint.endswith("/rerank") for endpoint in endpoints[1:])
    assert deleted == [("http://reranker/v1", "reranker")]


@patch("apps.documents.services.chunk_rerank_local_vllm.ordered_queryset_from_ids")
def test_malformed_cached_batch_response_falls_back_to_rerank_discovery(
    mock_ordered, monkeypatch
):
    capability = {
        "endpoint": "http://cached/score",
        "shape": "score_batch_text_pairs",
    }
    _configure_reranker(monkeypatch, capability=capability)
    monkeypatch.setattr(
        "apps.documents.services.chunk_rerank_local_vllm.rerank_model_is_qwen3_vl",
        lambda: False,
    )
    deleted: list[tuple[str, str]] = []
    monkeypatch.setattr(
        rag_cache,
        "delete_cached_rerank_capability",
        lambda base, model: deleted.append((base, model)),
    )
    mock_ordered.side_effect = lambda _model, identifiers: tuple(identifiers)
    rows = [MagicMock(pk=1, content="a"), MagicMock(pk=2, content="b")]
    endpoints: list[str] = []

    class MalformedResponse:
        status_code = 200

        @staticmethod
        def json():
            raise ValueError("malformed response")

    def post(endpoint, **_kwargs):
        endpoints.append(endpoint)
        if endpoint == capability["endpoint"]:
            return MalformedResponse()
        return _Response(200, {"results": [{"index": 1}]})

    monkeypatch.setattr(
        "apps.documents.services.chunk_rerank_local_vllm.requests.post", post
    )

    ranked = rerank_via_local_vllm(object, "query", rows, 1)

    assert ranked == (2,)
    assert endpoints[0] == "http://cached/score"
    assert any(endpoint.endswith("/rerank") for endpoint in endpoints[1:])
    assert deleted == [("http://reranker/v1", "reranker")]
