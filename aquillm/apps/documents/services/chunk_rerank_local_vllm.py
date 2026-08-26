"""Local vLLM / OpenAI-compatible rerank HTTP client."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from math import isfinite
from typing import TYPE_CHECKING, Any

import requests
import structlog

from apps.documents.services import rag_cache
from apps.documents.services.chunk_rerank_budget import (
    trim_rerank_pair,
)
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
    parse_score_results,
    parse_single_score,
)
from apps.documents.services.chunk_rerank_payload import rerank_document_payload
from lib.retrieval_redaction import RetrievalLogReason, retrieval_log_fields

if TYPE_CHECKING:
    from apps.documents.models.chunks import TextChunk

logger = structlog.stdlib.get_logger(__name__)


class _StrictCompleteScoringCapability:
    __slots__ = ()


_STRICT_COMPLETE_SCORING = _StrictCompleteScoringCapability()


def _is_complete_finite_scoring(
    pairs: list[tuple[int, float]],
    candidate_count: int,
) -> bool:
    return (
        len(pairs) == candidate_count
        and all(
            type(index) is int
            and 0 <= index < candidate_count
            and type(score) in (int, float)
            and isfinite(float(score))
            for index, score in pairs
        )
        and {index for index, _score in pairs} == set(range(candidate_count))
    )


def _score_one_document(
    *,
    endpoint: str,
    index: int,
    query: str,
    document: str,
    headers: dict[str, str],
    timeout: int,
    model_name: str,
    pair_token_limit: int,
    reserve_tokens: int,
) -> tuple[int, float] | None:
    try:
        request_query = query
        request_document = document
        adaptive_reserve = min(
            max(0, pair_token_limit - 2),
            max(reserve_tokens + 256, pair_token_limit // 2),
        )
        response = requests.post(
            endpoint,
            headers=headers,
            json={
                "model": model_name,
                "text_1": request_query,
                "text_2": request_document,
                # Let vLLM make the final fit with the model's actual tokenizer
                # and score template. The local cl100k estimate can otherwise
                # undercount a Qwen pair by enough to turn 1024 into 1025.
                "truncate_prompt_tokens": pair_token_limit,
                "truncation_side": "right",
            },
            timeout=timeout,
        )
        if response.status_code == 400 and request_document:
            retry_reserve = min(
                max(0, pair_token_limit - 2),
                max(adaptive_reserve + 256, (pair_token_limit * 3) // 4),
            )
            retry_query, retry_document = trim_rerank_pair(
                request_query,
                request_document,
                pair_token_limit,
                retry_reserve,
            )
            if (retry_query, retry_document) != (
                request_query,
                request_document,
            ):
                response = requests.post(
                    endpoint,
                    headers=headers,
                    json={
                        "model": model_name,
                        "text_1": retry_query,
                        "text_2": retry_document,
                        "truncate_prompt_tokens": pair_token_limit,
                        "truncation_side": "right",
                    },
                    timeout=timeout,
                )
        if response.status_code >= 400:
            return None
        score = parse_single_score(response.json())
        if type(score) not in (int, float) or not isfinite(float(score)):
            return None
        return index, float(score)
    except Exception:
        return None


def _score_documents_concurrently(
    *,
    endpoint: str,
    query: str,
    documents: list[str],
    headers: dict[str, str],
    timeout: int,
    max_workers: int,
    model_name: str,
    pair_token_limit: int,
    reserve_tokens: int,
) -> list[tuple[int, float]]:
    if not documents:
        return []
    workers = min(max(1, max_workers), len(documents))
    scores: list[tuple[int, float]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _score_one_document,
                endpoint=endpoint,
                index=index,
                query=query,
                document=document,
                headers=headers,
                timeout=timeout,
                model_name=model_name,
                pair_token_limit=pair_token_limit,
                reserve_tokens=reserve_tokens,
            )
            for index, document in enumerate(documents)
        ]
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                scores.append(result)
    return scores


def _rank_complete_scores(
    pairs: list[tuple[int, float]],
    chunks_list,
    top_k: int,
) -> list[int]:
    if not _is_complete_finite_scoring(pairs, len(chunks_list)):
        return []
    return [
        chunks_list[index].pk
        for index, _score in sorted(pairs, key=lambda item: item[1], reverse=True)[
            :top_k
        ]
    ]


def rerank_via_local_vllm(
    model_cls: type[TextChunk],
    query: str,
    chunks_list,
    top_k: int,
    *,
    _complete_scoring_capability: object | None = None,
):
    if (
        _complete_scoring_capability is not None
        and _complete_scoring_capability is not _STRICT_COMPLETE_SCORING
    ):
        raise PermissionError("complete local scoring requires its private capability")
    require_complete_scoring = _complete_scoring_capability is _STRICT_COMPLETE_SCORING
    if not chunks_list:
        return () if require_complete_scoring else model_cls.objects.none()

    base_v1 = rerank_base_url()
    model_name = rerank_model()
    query_signature = rag_cache.query_signature_for_rerank(query)
    candidate_ids = [chunk.pk for chunk in chunks_list]
    if not require_complete_scoring:
        cached_ranked = rag_cache.get_cached_rerank_result(
            query_signature,
            candidate_ids,
            top_k,
            model_name,
        )
        if cached_ranked:
            logger.info(
                "obs.rag.rerank_cache_hit",
                **retrieval_log_fields(
                    reason=RetrievalLogReason.COMPLETED,
                    count=0,
                    elapsed_ms=0.0,
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
    trimmed_pairs = [
        trim_rerank_pair(query, document, pair_limit, reserve_tokens)
        for document in raw_documents
    ]
    effective_query = trimmed_pairs[0][0]
    effective_documents = [document for _query, document in trimmed_pairs]

    multimodal_documents = [rerank_document_payload(chunk) for chunk in chunks_list]
    effective_multimodal_documents: list[Any] = []
    for mm_document, text_document in zip(
        multimodal_documents,
        effective_documents,
    ):
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
        if ranked_ids and not require_complete_scoring:
            rag_cache.set_cached_rerank_result(
                query_signature,
                candidate_ids,
                top_k,
                model_name,
                ranked_ids,
            )
            if capability:
                rag_cache.set_cached_rerank_capability(
                    base_v1,
                    model_name,
                    capability,
                )
        return ordered_queryset_from_ids(model_cls, ranked_ids)

    cached_capability = None
    if not require_complete_scoring:
        cached_capability = rag_cache.get_cached_rerank_capability(
            base_v1,
            model_name,
        )

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

    rerank_endpoints = [
        f"{base_root}/rerank",
        f"{base_root}/v2/rerank",
        f"{base_v1}/rerank",
    ]
    if cached_capability and cached_capability["shape"] == "rerank_documents":
        endpoint = cached_capability["endpoint"]
        rerank_endpoints = [endpoint] + [
            candidate for candidate in rerank_endpoints if candidate != endpoint
        ]
    if rerank_model_is_qwen3_vl() or require_complete_scoring:
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
        {
            "query": effective_query,
            "documents": effective_documents,
            "top_n": top_k,
        },
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

    observed_http_error = False
    for endpoint in rerank_endpoints:
        try:
            for payload in rerank_payloads:
                response = requests.post(
                    endpoint,
                    headers=headers,
                    json=payload,
                    timeout=timeout,
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
        endpoint = cached_capability["endpoint"]
        score_endpoints = [endpoint] + [
            candidate for candidate in score_endpoints if candidate != endpoint
        ]

    # Qwen3-VL accepts independent text pairs but rejects the list payload. Go
    # straight to the supported concurrent path instead of paying for a known 400.
    skip_batch_scores = rerank_model_is_qwen3_vl()
    for endpoint in score_endpoints:
        if not skip_batch_scores:
            batch_payloads: tuple[dict[str, Any], ...] = (
                {
                    "model": model_name,
                    "text_1": effective_query,
                    "text_2": effective_documents,
                },
                {
                    "text_1": effective_query,
                    "text_2": effective_documents,
                },
                {
                    "model": model_name,
                    "query": effective_query,
                    "documents": effective_documents,
                },
            )
            if has_multimodal_documents:
                batch_payloads += (
                    {
                        "model": model_name,
                        "query": [{"type": "text", "text": effective_query}],
                        "documents": effective_multimodal_documents,
                    },
                )
            try:
                for payload in batch_payloads:
                    response = requests.post(
                        endpoint,
                        headers=headers,
                        json=payload,
                        timeout=timeout,
                    )
                    if response.status_code in (404, 405):
                        continue
                    if response.status_code >= 400:
                        observed_http_error = True
                        continue
                    pairs = parse_score_results(response.json())
                    ranked_ids = _rank_complete_scores(pairs, chunks_list, top_k)
                    if ranked_ids:
                        return finish(
                            ranked_ids,
                            {
                                "endpoint": endpoint,
                                "shape": "score_batch_text_pairs",
                            },
                        )
            except Exception:
                pass

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
                reason=RetrievalLogReason.UPSTREAM_UNAVAILABLE,
                count=0,
                elapsed_ms=0.0,
            ),
        )

    return () if require_complete_scoring else model_cls.objects.none()


__all__ = ["_score_documents_concurrently", "rerank_via_local_vllm"]
