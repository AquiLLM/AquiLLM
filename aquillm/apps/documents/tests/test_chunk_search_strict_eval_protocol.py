"""Strict evaluation exercises the selected verified R1 protocol without fallback."""

from time import time

import pytest
from django.test import override_settings

from apps.documents.services.rerank_transport_policy import RerankProtocolV1
from apps.documents.tests.test_chunk_search_graph_overlay import _chunk


def _verified(monkeypatch, mode):
    binding = RerankProtocolV1(
        "http://local/score",
        "score_batch_text_pairs" if mode == "batch" else "score_single_text_pair",
        "Qwen/reranker",
        time() - 1,
        time() + 300,
    )
    monkeypatch.setattr(
        "apps.documents.services.rag_cache.get_cached_rerank_capability",
        lambda *args, **kwargs: binding.as_cache_entry(),
    )


@pytest.mark.parametrize("partial_mode", ("batch", "per_document"))
@override_settings(RAG_CACHE_ENABLED=False)
def test_strict_eval_rejects_partial_scoring_even_when_top_k_is_full(
    monkeypatch,
    partial_mode,
):
    from apps.documents.services import chunk_rerank
    from apps.documents.services import chunk_rerank_local_vllm as local_vllm

    rows = tuple(_chunk(identifier) for identifier in range(1, 13))
    observed_modes = set()
    batch_calls = []

    class Response:
        status_code = 200

        def __init__(self, payload):
            self.payload = payload

        def json(self):
            return self.payload

    def post(_url, *, json, **_kwargs):
        documents = json.get("text_2", json.get("documents"))
        if isinstance(documents, list):
            observed_modes.add("batch")
            count = 6 if not batch_calls else 4
            batch_calls.append(len(documents))
            return Response(
                {
                    "data": [
                        {"index": index, "score": float(10 - index)}
                        for index in range(count)
                    ]
                }
            )
        observed_modes.add("per_document")
        identifier = int(str(documents).rsplit("-", 1)[1])
        if partial_mode == "per_document" and identifier <= 10:
            return Response({"score": float(13 - identifier)})
        return Response({})

    def ordered(_model, identifiers):
        by_id = {row.pk: row for row in rows}
        return tuple(by_id[identifier] for identifier in identifiers)

    monkeypatch.setenv("APP_RERANK_PROVIDER", "local")
    monkeypatch.setattr(local_vllm, "rerank_base_url", lambda: "http://local/v1")
    monkeypatch.setattr(local_vllm, "rerank_model", lambda: "Qwen/reranker")
    monkeypatch.setattr(
        local_vllm, "rerank_model_is_qwen3_vl", lambda: partial_mode == "per_document"
    )
    monkeypatch.setattr(local_vllm, "rerank_document_payload", lambda row: row.content)
    monkeypatch.setattr(local_vllm, "ordered_queryset_from_ids", ordered)
    monkeypatch.setattr(local_vllm.requests, "post", post)

    _verified(monkeypatch, partial_mode)
    with pytest.raises(chunk_rerank.StrictRerankUnavailable, match="empty"):
        chunk_rerank._strict_local_rerank_chunks(
            object,
            "query",
            rows,
            10,
            _capability=chunk_rerank._STRICT_EVALUATION_RERANK,
        )
    assert observed_modes == {partial_mode}
    if partial_mode == "batch":
        assert batch_calls == [6, 6]


@pytest.mark.parametrize("score_mode", ("batch", "per_document"))
@pytest.mark.parametrize("bad_score", (True, float("nan"), float("inf"), -float("inf")))
@override_settings(RAG_CACHE_ENABLED=False)
def test_strict_eval_rejects_nonfinite_or_boolean_scores(
    monkeypatch,
    score_mode,
    bad_score,
):
    from apps.documents.services import chunk_rerank
    from apps.documents.services import chunk_rerank_local_vllm as local_vllm

    rows = (_chunk(1), _chunk(2))
    observed_modes = set()

    class Response:
        status_code = 200

        def __init__(self, payload):
            self.payload = payload

        def json(self):
            return self.payload

    def post(_url, *, json, **_kwargs):
        documents = json.get("text_2", json.get("documents"))
        if isinstance(documents, list):
            observed_modes.add("batch")
            return Response(
                {"data": [{"index": 0, "score": 1.0}, {"index": 1, "score": bad_score}]}
            )
        observed_modes.add("per_document")
        if score_mode == "batch":
            return Response({})
        return Response({"score": 1.0 if str(documents).endswith("-1") else bad_score})

    monkeypatch.setenv("APP_RERANK_PROVIDER", "local")
    monkeypatch.setattr(local_vllm, "rerank_base_url", lambda: "http://local/v1")
    monkeypatch.setattr(local_vllm, "rerank_model", lambda: "Qwen/reranker")
    monkeypatch.setattr(
        local_vllm, "rerank_model_is_qwen3_vl", lambda: score_mode == "per_document"
    )
    monkeypatch.setattr(local_vllm, "rerank_document_payload", lambda row: row.content)
    monkeypatch.setattr(
        local_vllm,
        "ordered_queryset_from_ids",
        lambda _model, identifiers: tuple(rows[index - 1] for index in identifiers),
    )
    monkeypatch.setattr(local_vllm.requests, "post", post)

    _verified(monkeypatch, score_mode)
    with pytest.raises(chunk_rerank.StrictRerankUnavailable, match="empty"):
        chunk_rerank._strict_local_rerank_chunks(
            object,
            "query",
            rows,
            2,
            _capability=chunk_rerank._STRICT_EVALUATION_RERANK,
        )
    assert observed_modes == {score_mode}
