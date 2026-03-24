"""Rerank HTTP client result and capability caches."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from django.test import override_settings

from apps.documents.services.chunk_rerank_local_vllm import rerank_via_local_vllm


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
            rerank_via_local_vllm(MC, "user query must not appear in metrics log", chunks, 2)

    mock_post.assert_not_called()
    hit_calls = [
        c
        for c in mock_logger.info.call_args_list
        if c.args and "cache_hit=1" in c.args[0]
    ]
    assert hit_calls
    joined = " ".join(str(c.args) for c in hit_calls)
    assert "secret-query-text" not in joined
    assert "user query must not appear" not in joined
