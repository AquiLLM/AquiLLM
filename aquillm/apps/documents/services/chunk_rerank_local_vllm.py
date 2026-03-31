"""Local vLLM / OpenAI-compatible rerank HTTP client."""
from __future__ import annotations

import structlog
from typing import Any, Type, TYPE_CHECKING

import requests

from apps.documents.services import rag_cache
from apps.documents.services.chunk_rerank_config import (
    rerank_api_key,
    rerank_base_url,
    rerank_doc_char_limit,
    rerank_model,
    rerank_model_is_qwen3_vl,
    rerank_timeout_seconds,
)
from apps.documents.services.chunk_rerank_parse import (
    ordered_queryset_from_ids,
    parse_rerank_results,
    parse_score_results,
    parse_single_score,
)
from apps.documents.services.chunk_rerank_payload import rerank_document_payload

if TYPE_CHECKING:
    from apps.documents.models.chunks import TextChunk

logger = structlog.stdlib.get_logger(__name__)


def rerank_via_local_vllm(model_cls: Type["TextChunk"], query: str, chunks_list, top_k: int):
    if not chunks_list:
        return model_cls.objects.none()

    base_v1 = rerank_base_url()
    rm = rerank_model()
    qsig = rag_cache.query_signature_for_rerank(query)
    cand_ids = [c.pk for c in chunks_list]
    cached_ranked = rag_cache.get_cached_rerank_result(qsig, cand_ids, top_k, rm)
    if cached_ranked:
        logger.info(
            "obs.rag.rerank_local_cache_hit",
            candidates=len(cand_ids),
            top_k=top_k,
        )
        return ordered_queryset_from_ids(model_cls, cached_ranked)

    raw_documents = [chunk.content for chunk in chunks_list]
    multimodal_documents = [rerank_document_payload(chunk) for chunk in chunks_list]
    char_limit = rerank_doc_char_limit()
    documents = [doc[:char_limit] if len(doc) > char_limit else doc for doc in raw_documents]
    multimodal_documents_trimmed: list[Any] = []
    for mm_doc, text_doc in zip(multimodal_documents, documents):
        if isinstance(mm_doc, list):
            normalized: list[dict[str, Any]] = []
            for part in mm_doc:
                if isinstance(part, dict) and part.get("type") == "text":
                    normalized.append({"type": "text", "text": text_doc})
                else:
                    normalized.append(part)
            multimodal_documents_trimmed.append(normalized)
        else:
            multimodal_documents_trimmed.append(text_doc)
    has_multimodal_docs = any(isinstance(doc, list) for doc in multimodal_documents_trimmed)
    if not documents:
        return model_cls.objects.none()

    base_root = base_v1[:-3] if base_v1.endswith("/v1") else base_v1

    def _finish(ranked_ids: list[int], winning_endpoint: str | None = None):
        if ranked_ids:
            rag_cache.set_cached_rerank_result(qsig, cand_ids, top_k, rm, ranked_ids)
            if winning_endpoint:
                rag_cache.set_cached_rerank_capability(base_v1, rm, winning_endpoint)
        return ordered_queryset_from_ids(model_cls, ranked_ids)

    headers = {"Content-Type": "application/json"}
    api_key = rerank_api_key()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    timeout = rerank_timeout_seconds()
    first_http_error: str | None = None

    effective_query = query
    effective_documents = documents
    effective_multimodal_documents = multimodal_documents_trimmed

    def _track_http_error(endpoint: str, response: requests.Response):
        nonlocal first_http_error
        if first_http_error is not None:
            return
        response_body = ""
        try:
            response_body = response.text[:600]
        except Exception:
            response_body = "<unavailable>"
        first_http_error = f"{endpoint} -> {response.status_code}: {response_body}"

    rerank_endpoints = [
        f"{base_root}/rerank",
        f"{base_root}/v2/rerank",
        f"{base_v1}/rerank",
    ]
    preferred = rag_cache.get_cached_rerank_capability(base_v1, rm)
    if preferred:
        rerank_endpoints = [preferred] + [e for e in rerank_endpoints if e != preferred]
    if rerank_model_is_qwen3_vl():
        rerank_endpoints = []
    rerank_payloads = (
        {
            "model": rerank_model(),
            "query": effective_query,
            "documents": effective_documents,
            "top_n": top_k,
        },
        {
            "model": rerank_model(),
            "query": effective_query,
            "documents": [{"text": doc} for doc in effective_documents],
            "top_n": top_k,
        },
        {
            "query": effective_query,
            "documents": effective_documents,
            "top_n": top_k,
        },
    )
    if has_multimodal_docs:
        rerank_payloads = rerank_payloads + (
            {
                "model": rerank_model(),
                "query": effective_query,
                "documents": effective_multimodal_documents,
                "top_n": top_k,
            },
        )

    for endpoint in rerank_endpoints:
        try:
            for rerank_payload in rerank_payloads:
                response = requests.post(
                    endpoint, headers=headers, json=rerank_payload, timeout=timeout
                )
                if response.status_code in (404, 405):
                    continue
                if response.status_code >= 400:
                    _track_http_error(endpoint, response)
                    continue
                ranked_ids = parse_rerank_results(response.json(), chunks_list)
                if ranked_ids:
                    return _finish(ranked_ids, endpoint)
        except Exception:
            continue

    for endpoint in (f"{base_root}/score", f"{base_v1}/score"):
        try:
            batch_payloads = (
                {
                    "model": rerank_model(),
                    "text_1": effective_query,
                    "text_2": effective_documents,
                },
                {
                    "text_1": effective_query,
                    "text_2": effective_documents,
                },
                {
                    "model": rerank_model(),
                    "query": effective_query,
                    "documents": effective_documents,
                },
            )
            if has_multimodal_docs:
                batch_payloads = batch_payloads + (
                    {
                        "model": rerank_model(),
                        "query": [{"type": "text", "text": effective_query}],
                        "documents": effective_multimodal_documents,
                    },
                    {
                        "model": rerank_model(),
                        "text_1": [{"type": "text", "text": effective_query}],
                        "text_2": effective_multimodal_documents,
                    },
                )
            for batch_payload in batch_payloads:
                response = requests.post(
                    endpoint, headers=headers, json=batch_payload, timeout=timeout
                )
                if response.status_code in (404, 405):
                    continue
                if response.status_code >= 400:
                    _track_http_error(endpoint, response)
                    continue
                pairs = parse_score_results(response.json())
                if not pairs:
                    continue
                ranked_ids = [
                    chunks_list[idx].pk
                    for idx, _ in sorted(pairs, key=lambda item: item[1], reverse=True)[:top_k]
                ]
                if ranked_ids:
                    return _finish(ranked_ids, endpoint)

            scored: list[tuple[int, float]] = []
            for idx, doc in enumerate(effective_documents):
                mm_doc = effective_multimodal_documents[idx]
                score_payloads = (
                    {
                        "model": rerank_model(),
                        "text_1": effective_query,
                        "text_2": doc,
                    },
                    {
                        "model": rerank_model(),
                        "query": effective_query,
                        "document": doc,
                    },
                    {
                        "text_1": effective_query,
                        "text_2": doc,
                    },
                )
                if has_multimodal_docs and isinstance(mm_doc, list):
                    score_payloads = score_payloads + (
                        {
                            "model": rerank_model(),
                            "query": [{"type": "text", "text": effective_query}],
                            "document": mm_doc,
                        },
                        {
                            "model": rerank_model(),
                            "text_1": [{"type": "text", "text": effective_query}],
                            "text_2": mm_doc,
                        },
                        {
                            "model": rerank_model(),
                            "messages": [
                                {
                                    "role": "system",
                                    "content": "Given a search query, retrieve relevant candidates that answer the query.",
                                },
                                {"role": "query", "content": effective_query},
                                {"role": "document", "content": mm_doc},
                            ],
                        },
                    )
                score_found = False
                for score_payload in score_payloads:
                    response = requests.post(
                        endpoint, headers=headers, json=score_payload, timeout=timeout
                    )
                    if response.status_code in (404, 405):
                        continue
                    if response.status_code >= 400:
                        _track_http_error(endpoint, response)
                        continue
                    score = parse_single_score(response.json())
                    scored.append((idx, score))
                    score_found = True
                    break
                if not score_found:
                    continue

            if not scored:
                continue

            scored.sort(key=lambda item: item[1], reverse=True)
            ranked_ids = [chunks_list[idx].pk for idx, _ in scored[:top_k]]
            if ranked_ids:
                return _finish(ranked_ids, endpoint)
        except Exception:
            continue

    if first_http_error:
        logger.warning("obs.rag.rerank_local_all_failed", first_error=first_http_error)

    return model_cls.objects.none()


__all__ = ["rerank_via_local_vllm"]
