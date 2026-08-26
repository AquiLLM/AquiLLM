"""Token-budget contracts for local reranker query/document pairs."""

from __future__ import annotations

from apps.documents.services.chunk_rerank_budget import (
    count_rerank_tokens,
    trim_rerank_pair,
)
from apps.documents.services.chunk_rerank_config import (
    rerank_template_reserve_tokens,
)
from apps.documents.services.chunk_rerank_local_vllm import _score_one_document


def test_trim_rerank_pair_respects_token_budget_and_preserves_normal_query():
    query = "attensity calibration"
    document = "evidence " * 4000

    trimmed_query, trimmed_document = trim_rerank_pair(
        query,
        document,
        max_pair_tokens=900,
        reserve_tokens=64,
    )

    assert trimmed_query == query
    assert count_rerank_tokens(trimmed_query, trimmed_document) <= 836
    assert trimmed_document
    assert len(trimmed_document) < len(document)


def test_trim_rerank_pair_is_deterministic_for_unicode_academic_text():
    first = trim_rerank_pair(
        "β calibration",
        "λ evidence " * 2000,
        max_pair_tokens=900,
        reserve_tokens=64,
    )
    second = trim_rerank_pair(
        "β calibration",
        "λ evidence " * 2000,
        max_pair_tokens=900,
        reserve_tokens=64,
    )

    assert first == second


def test_oversized_query_leaves_room_for_document_evidence():
    trimmed_query, trimmed_document = trim_rerank_pair(
        "query " * 2000,
        "document evidence " * 2000,
        max_pair_tokens=300,
        reserve_tokens=40,
    )

    assert trimmed_query
    assert trimmed_document
    assert count_rerank_tokens(trimmed_query, trimmed_document) <= 260


def test_rerank_template_reserve_defaults_to_qwen_safe_margin(monkeypatch):
    monkeypatch.delenv("APP_RERANK_TEMPLATE_RESERVE_TOKENS", raising=False)

    assert rerank_template_reserve_tokens() == 256


def test_single_score_retries_context_overflow_with_tighter_pair(monkeypatch):
    payloads: list[dict] = []

    class Response:
        def __init__(self, status_code: int, score: float | None = None):
            self.status_code = status_code
            self._score = score

        def json(self):
            return {"score": self._score}

    def post(_endpoint, *, json, **_kwargs):
        payloads.append(json)
        if len(payloads) == 1:
            return Response(400)
        return Response(200, 0.75)

    monkeypatch.setattr(
        "apps.documents.services.chunk_rerank_local_vllm.requests.post",
        post,
    )
    monkeypatch.setattr(
        "apps.documents.services.chunk_rerank_local_vllm.parse_single_score",
        lambda payload: payload["score"],
    )

    result = _score_one_document(
        endpoint="http://reranker/score",
        index=7,
        query="attensity calibration",
        document="dense evidence " * 1000,
        headers={},
        timeout=10,
        model_name="Qwen/Qwen3-VL-Reranker-2B",
        pair_token_limit=1024,
        reserve_tokens=256,
    )

    assert result == (7, 0.75)
    assert len(payloads) == 2
    assert len(payloads[1]["text_2"]) < len(payloads[0]["text_2"])
    assert count_rerank_tokens(
        payloads[1]["text_1"], payloads[1]["text_2"]
    ) <= 256


def test_single_score_preserves_locally_budgeted_evidence_for_server_fit(monkeypatch):
    payloads: list[dict] = []

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {"score": 0.9}

    def post(_endpoint, *, json, **_kwargs):
        payloads.append(json)
        return Response()

    monkeypatch.setattr(
        "apps.documents.services.chunk_rerank_local_vllm.requests.post",
        post,
    )

    query, document = trim_rerank_pair(
        "attensity calibration",
        "dense evidence " * 1000,
        max_pair_tokens=1024,
        reserve_tokens=256,
    )
    assert count_rerank_tokens(query, document) == 768

    result = _score_one_document(
        endpoint="http://reranker/score",
        index=3,
        query=query,
        document=document,
        headers={},
        timeout=10,
        model_name="Qwen/Qwen3-VL-Reranker-2B",
        pair_token_limit=1024,
        reserve_tokens=256,
    )

    assert result == (3, 0.9)
    assert len(payloads) == 1
    assert payloads[0]["text_1"] == query
    assert payloads[0]["text_2"] == document
    assert payloads[0]["truncate_prompt_tokens"] == 1024
    assert payloads[0]["truncation_side"] == "right"


def test_single_score_uses_server_tokenizer_at_estimator_boundary(monkeypatch):
    payloads: list[dict] = []

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {"score": 0.8}

    def post(_endpoint, *, json, **_kwargs):
        payloads.append(json)
        return Response()

    query, document = trim_rerank_pair(
        "attensity calibration",
        "dense evidence " * 1000,
        max_pair_tokens=1024,
        reserve_tokens=257,
    )
    assert count_rerank_tokens(query, document) == 767

    monkeypatch.setattr(
        "apps.documents.services.chunk_rerank_local_vllm.requests.post",
        post,
    )

    result = _score_one_document(
        endpoint="http://reranker/score",
        index=4,
        query=query,
        document=document,
        headers={},
        timeout=10,
        model_name="Qwen/Qwen3-VL-Reranker-2B",
        pair_token_limit=1024,
        reserve_tokens=256,
    )

    assert result == (4, 0.8)
    assert len(payloads) == 1
    assert payloads[0]["text_1"] == query
    assert payloads[0]["text_2"] == document
    assert payloads[0]["truncate_prompt_tokens"] == 1024
    assert payloads[0]["truncation_side"] == "right"
