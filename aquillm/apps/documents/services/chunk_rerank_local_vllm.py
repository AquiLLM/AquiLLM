"""Local vLLM / OpenAI-compatible rerank HTTP client."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import requests
import structlog
from lib.retrieval_redaction import RetrievalLogReason, retrieval_log_fields

from apps.documents.services import chunk_rerank_score, rag_cache
from apps.documents.services.chunk_rerank_budget import trim_rerank_documents
from apps.documents.services.chunk_rerank_config import (
    rerank_api_key,
    rerank_base_url,
    rerank_doc_char_limit,
    rerank_model,
    rerank_model_is_qwen3_vl,
    rerank_pair_token_limit,
    rerank_score_concurrency,
    rerank_template_reserve_tokens,
    rerank_timeout_seconds,
)
from apps.documents.services.chunk_rerank_parse import (
    ordered_queryset_from_ids,
    parse_rerank_results,
)
from apps.documents.services.chunk_rerank_payload import rerank_document_payload

if TYPE_CHECKING:
    from apps.documents.models.chunks import TextChunk

logger = structlog.stdlib.get_logger(__name__)
_batch_score_payloads = chunk_rerank_score.batch_score_payloads
_parse_batch_scores = chunk_rerank_score.parse_batch_scores
_rank_complete_scores = chunk_rerank_score.rank_complete_scores
_score_documents_concurrently = chunk_rerank_score.score_documents_concurrently
_score_one_document = chunk_rerank_score.score_one_document


def rerank_via_local_vllm(
    model_cls: type[TextChunk], query: str, chunks_list, top_k: int
):
    if not chunks_list:
        return model_cls.objects.none()

    base_v1 = rerank_base_url()
    model_name = rerank_model()
    query_signature = rag_cache.query_signature_for_rerank(query)
    candidate_ids = [chunk.pk for chunk in chunks_list]
    cached_ranked = rag_cache.get_cached_rerank_result(
        query_signature, candidate_ids, top_k, model_name
    )
    if cached_ranked:
        logger.info(
            "obs.rag.rerank_cache_hit",
            **retrieval_log_fields(
                reason=RetrievalLogReason.COMPLETED, count=0, elapsed_ms=0.0
            ),
        )
        return ordered_queryset_from_ids(model_cls, cached_ranked)

    char_limit = rerank_doc_char_limit()
    raw_documents = [
        chunk.content[:char_limit] if len(chunk.content) > char_limit else chunk.content
        for chunk in chunks_list
    ]
    pair_limit = rerank_pair_token_limit()
    reserve_tokens = rerank_template_reserve_tokens()
    effective_query, effective_documents = trim_rerank_documents(
        query, raw_documents, pair_limit, reserve_tokens
    )

    multimodal_documents = [rerank_document_payload(chunk) for chunk in chunks_list]
    effective_multimodal_documents: list[Any] = []
    for mm_document, text_document in zip(multimodal_documents, effective_documents):
        if isinstance(mm_document, list):
            normalized: list[dict[str, Any]] = []
            for part in mm_document:
                if isinstance(part, dict) and part.get("type") == "text":
                    normalized.append({"type": "text", "text": text_document})
                else:
                    normalized.append(part)
            effective_multimodal_documents.append(normalized)
        else:
            effective_multimodal_documents.append(text_document)
    has_multimodal_documents = any(
        isinstance(document, list) for document in effective_multimodal_documents
    )

    base_root = base_v1[:-3] if base_v1.endswith("/v1") else base_v1
    headers = {"Content-Type": "application/json"}
    api_key = rerank_api_key()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    timeout = rerank_timeout_seconds()

    def finish(
        ranked_ids: list[int],
        capability: rag_cache.RerankCapability | None = None,
    ):
        if ranked_ids:
            rag_cache.set_cached_rerank_result(
                query_signature, candidate_ids, top_k, model_name, ranked_ids
            )
            if capability:
                rag_cache.set_cached_rerank_capability(base_v1, model_name, capability)
        return ordered_queryset_from_ids(model_cls, ranked_ids)

    observed_http_error = False
    batch_payloads = _batch_score_payloads(
        model_name,
        effective_query,
        effective_documents,
        effective_multimodal_documents,
        has_multimodal_documents,
    )

    def score_batch(endpoint: str) -> list[int]:
        nonlocal observed_http_error
        for payload in batch_payloads:
            try:
                response = requests.post(
                    endpoint, headers=headers, json=payload, timeout=timeout
                )
                if response.status_code in (404, 405):
                    continue
                if response.status_code >= 400:
                    observed_http_error = True
                    continue
                ranked_ids = _parse_batch_scores(response.json(), chunks_list, top_k)
            except Exception:
                observed_http_error = True
                continue
            if ranked_ids:
                return ranked_ids
        return []

    cached_capability = rag_cache.get_cached_rerank_capability(base_v1, model_name)
    if cached_capability and cached_capability["shape"] == "score_single_text_pair":
        scores = _score_documents_concurrently(
            endpoint=cached_capability["endpoint"],
            query=effective_query,
            documents=effective_documents,
            headers=headers,
            timeout=timeout,
            max_workers=rerank_score_concurrency(),
            model_name=model_name,
            pair_token_limit=pair_limit,
            reserve_tokens=reserve_tokens,
        )
        ranked_ids = _rank_complete_scores(scores, chunks_list, top_k)
        if ranked_ids:
            return finish(ranked_ids, cached_capability)
        rag_cache.delete_cached_rerank_capability(base_v1, model_name)
        cached_capability = None
    elif cached_capability and cached_capability["shape"] == "score_batch_text_pairs":
        ranked_ids = score_batch(cached_capability["endpoint"])
        if ranked_ids:
            return finish(ranked_ids, cached_capability)
        rag_cache.delete_cached_rerank_capability(base_v1, model_name)
        cached_capability = None

    rerank_endpoints = [
        f"{base_root}/rerank",
        f"{base_root}/v2/rerank",
        f"{base_v1}/rerank",
    ]
    if cached_capability and cached_capability["shape"] == "rerank_documents":
        preferred = cached_capability["endpoint"]
        rerank_endpoints = [preferred] + [
            endpoint for endpoint in rerank_endpoints if endpoint != preferred
        ]
    if rerank_model_is_qwen3_vl():
        rerank_endpoints = []

    rerank_payloads: tuple[dict[str, Any], ...] = (
        {
            "model": model_name,
            "query": effective_query,
            "documents": effective_documents,
            "top_n": top_k,
        },
        {
            "model": model_name,
            "query": effective_query,
            "documents": [{"text": doc} for doc in effective_documents],
            "top_n": top_k,
        },
        {"query": effective_query, "documents": effective_documents, "top_n": top_k},
    )
    if has_multimodal_documents:
        rerank_payloads += (
            {
                "model": model_name,
                "query": effective_query,
                "documents": effective_multimodal_documents,
                "top_n": top_k,
            },
        )

    for endpoint in rerank_endpoints:
        try:
            for payload in rerank_payloads:
                response = requests.post(
                    endpoint, headers=headers, json=payload, timeout=timeout
                )
                if response.status_code in (404, 405):
                    continue
                if response.status_code >= 400:
                    observed_http_error = True
                    continue
                ranked_ids = parse_rerank_results(response.json(), chunks_list)
                if ranked_ids:
                    return finish(
                        ranked_ids,
                        {"endpoint": endpoint, "shape": "rerank_documents"},
                    )
        except Exception:
            continue

    score_endpoints = [f"{base_root}/score", f"{base_v1}/score"]
    if cached_capability and cached_capability["shape"] == "score_batch_text_pairs":
        preferred = cached_capability["endpoint"]
        score_endpoints = [preferred] + [
            endpoint for endpoint in score_endpoints if endpoint != preferred
        ]

    skip_batch_scores = rerank_model_is_qwen3_vl()
    for endpoint in score_endpoints:
        if not skip_batch_scores:
            ranked_ids = score_batch(endpoint)
            if ranked_ids:
                return finish(
                    ranked_ids,
                    {"endpoint": endpoint, "shape": "score_batch_text_pairs"},
                )

        scores = _score_documents_concurrently(
            endpoint=endpoint,
            query=effective_query,
            documents=effective_documents,
            headers=headers,
            timeout=timeout,
            max_workers=rerank_score_concurrency(),
            model_name=model_name,
            pair_token_limit=pair_limit,
            reserve_tokens=reserve_tokens,
        )
        ranked_ids = _rank_complete_scores(scores, chunks_list, top_k)
        if ranked_ids:
            return finish(
                ranked_ids,
                {"endpoint": endpoint, "shape": "score_single_text_pair"},
            )
        observed_http_error = True

    if observed_http_error:
        logger.warning(
            "obs.rag.rerank_requests_failed",
            **retrieval_log_fields(
                reason=RetrievalLogReason.UPSTREAM_UNAVAILABLE, count=0, elapsed_ms=0.0
            ),
        )
    return model_cls.objects.none()


__all__ = [
    "_score_documents_concurrently",
    "_score_one_document",
    "rerank_via_local_vllm",
]
