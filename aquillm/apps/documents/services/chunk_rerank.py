"""Reranking for vector/trigram chunk candidates (local vLLM, Cohere, fallbacks)."""
from __future__ import annotations

import logging
from os import getenv
from typing import TYPE_CHECKING, Any, Type

import requests
from django.apps import apps
from django.db.models import Case, When

from apps.documents.services.chunk_embeddings import image_data_url, multimodal_caption

if TYPE_CHECKING:
    from apps.documents.models.chunks import TextChunk

logger = logging.getLogger(__name__)


def rerank_document_payload(chunk: TextChunk) -> Any:
    if chunk.modality != chunk.Modality.IMAGE:
        return chunk.content
    data_url = image_data_url(chunk)
    if not data_url:
        return chunk.content
    caption = multimodal_caption(chunk)
    return [
        {"type": "text", "text": caption},
        {"type": "image_url", "image_url": {"url": data_url}},
    ]


def _fallback_rerank(model_cls: Type[TextChunk], chunks, top_k: int):
    chunk_ids = [chunk.pk for chunk in list(chunks)[:top_k]]
    if not chunk_ids:
        return model_cls.objects.none()
    preserved = Case(*[When(pk=pk, then=pos) for pos, pk in enumerate(chunk_ids)])
    return model_cls.objects.filter(pk__in=chunk_ids).order_by(preserved)


def _rerank_base_url() -> str:
    base_url = (
        getenv("APP_RERANK_BASE_URL")
        or getenv("VLLM_RERANK_BASE_URL")
        or getenv("VLLM_BASE_URL")
        or "http://vllm_rerank:8000/v1"
    ).rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    return base_url


def _rerank_api_key() -> str:
    return (
        getenv("APP_RERANK_API_KEY")
        or getenv("VLLM_RERANK_API_KEY")
        or getenv("VLLM_API_KEY")
        or "EMPTY"
    )


def _rerank_model() -> str:
    return (
        getenv("APP_RERANK_MODEL")
        or getenv("VLLM_RERANK_MODEL")
        or "Qwen/Qwen3-Reranker-4B"
    )


def _rerank_timeout_seconds() -> int:
    try:
        timeout = int((getenv("APP_RERANK_TIMEOUT_SECONDS") or "3").strip())
    except Exception:
        timeout = 3
    return timeout if timeout > 0 else 3


def _ordered_queryset_from_ids(model_cls: Type[TextChunk], ranked_ids: list[int]):
    if not ranked_ids:
        return model_cls.objects.none()
    preserved = Case(*[When(pk=pk, then=pos) for pos, pk in enumerate(ranked_ids)])
    return model_cls.objects.filter(pk__in=ranked_ids).order_by(preserved)


def _parse_rerank_results(body, chunks_list) -> list[int]:
    results = []
    if isinstance(body, dict):
        if isinstance(body.get("results"), list):
            results = body.get("results", [])
        elif isinstance(body.get("data"), list):
            results = body.get("data", [])
    ranked_ids: list[int] = []
    seen: set[int] = set()
    for result in results:
        idx = result.get("index") if isinstance(result, dict) else None
        if not isinstance(idx, int) or idx < 0 or idx >= len(chunks_list):
            continue
        chunk_pk = chunks_list[idx].pk
        if chunk_pk in seen:
            continue
        seen.add(chunk_pk)
        ranked_ids.append(chunk_pk)
    return ranked_ids


def _parse_score_results(body) -> list[tuple[int, float]]:
    pairs: list[tuple[int, float]] = []
    if not isinstance(body, dict):
        return pairs
    raw_items = None
    if isinstance(body.get("data"), list):
        raw_items = body.get("data")
    elif isinstance(body.get("results"), list):
        raw_items = body.get("results")
    if not raw_items:
        return pairs
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        idx = item.get("index")
        if not isinstance(idx, int):
            continue
        score = item.get("score")
        if not isinstance(score, (int, float)):
            score = item.get("relevance_score")
        if not isinstance(score, (int, float)):
            continue
        pairs.append((idx, float(score)))
    return pairs


def _parse_single_score(body) -> float:
    if isinstance(body, (int, float)):
        return float(body)
    if isinstance(body, dict):
        if isinstance(body.get("score"), (int, float)):
            return float(body["score"])
        data = body.get("data")
        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict):
                for key in ("score", "relevance_score"):
                    if isinstance(first.get(key), (int, float)):
                        return float(first[key])
        results = body.get("results")
        if isinstance(results, list) and results:
            first = results[0]
            if isinstance(first, dict):
                for key in ("score", "relevance_score"):
                    if isinstance(first.get(key), (int, float)):
                        return float(first[key])
    raise ValueError(f"Unable to parse score response: {body!r}")


def _rerank_model_is_qwen3_vl() -> bool:
    model_name = (_rerank_model() or "").lower()
    return "qwen3-vl-reranker" in model_name


def _rerank_doc_char_limit() -> int:
    raw = (getenv("APP_RERANK_DOC_CHAR_LIMIT") or "").strip()
    if not raw:
        return 2000
    try:
        value = int(raw)
        return value if value > 0 else 2000
    except Exception:
        return 2000


def _rerank_via_local_vllm(model_cls: Type[TextChunk], query: str, chunks_list, top_k: int):
    raw_documents = [chunk.content for chunk in chunks_list]
    multimodal_documents = [rerank_document_payload(chunk) for chunk in chunks_list]
    char_limit = _rerank_doc_char_limit()
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

    headers = {"Content-Type": "application/json"}
    api_key = _rerank_api_key()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    base_v1 = _rerank_base_url()
    base_root = base_v1[:-3] if base_v1.endswith("/v1") else base_v1
    timeout = _rerank_timeout_seconds()
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

    rerank_endpoints = (
        f"{base_root}/rerank",
        f"{base_root}/v2/rerank",
        f"{base_v1}/rerank",
    )
    if _rerank_model_is_qwen3_vl():
        rerank_endpoints = ()
    rerank_payloads = (
        {
            "model": _rerank_model(),
            "query": effective_query,
            "documents": effective_documents,
            "top_n": top_k,
        },
        {
            "model": _rerank_model(),
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
                "model": _rerank_model(),
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
                ranked_ids = _parse_rerank_results(response.json(), chunks_list)
                if ranked_ids:
                    return _ordered_queryset_from_ids(model_cls, ranked_ids)
        except Exception:
            continue

    for endpoint in (f"{base_root}/score", f"{base_v1}/score"):
        try:
            batch_payloads = (
                {
                    "model": _rerank_model(),
                    "text_1": effective_query,
                    "text_2": effective_documents,
                },
                {
                    "text_1": effective_query,
                    "text_2": effective_documents,
                },
                {
                    "model": _rerank_model(),
                    "query": effective_query,
                    "documents": effective_documents,
                },
            )
            if has_multimodal_docs:
                batch_payloads = batch_payloads + (
                    {
                        "model": _rerank_model(),
                        "query": [{"type": "text", "text": effective_query}],
                        "documents": effective_multimodal_documents,
                    },
                    {
                        "model": _rerank_model(),
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
                pairs = _parse_score_results(response.json())
                if not pairs:
                    continue
                ranked_ids = [
                    chunks_list[idx].pk
                    for idx, _ in sorted(pairs, key=lambda item: item[1], reverse=True)[:top_k]
                ]
                if ranked_ids:
                    return _ordered_queryset_from_ids(model_cls, ranked_ids)

            scored: list[tuple[int, float]] = []
            for idx, doc in enumerate(effective_documents):
                mm_doc = effective_multimodal_documents[idx]
                score_payloads = (
                    {
                        "model": _rerank_model(),
                        "text_1": effective_query,
                        "text_2": doc,
                    },
                    {
                        "model": _rerank_model(),
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
                            "model": _rerank_model(),
                            "query": [{"type": "text", "text": effective_query}],
                            "document": mm_doc,
                        },
                        {
                            "model": _rerank_model(),
                            "text_1": [{"type": "text", "text": effective_query}],
                            "text_2": mm_doc,
                        },
                        {
                            "model": _rerank_model(),
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
                    score = _parse_single_score(response.json())
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
                return _ordered_queryset_from_ids(model_cls, ranked_ids)
        except Exception:
            continue

    if first_http_error:
        logger.warning("All local rerank requests failed. First error: %s", first_http_error)

    return model_cls.objects.none()


def rerank_chunks(model_cls: Type[TextChunk], query: str, chunks, top_k: int):
    chunks_list = list(chunks)
    provider = (getenv("APP_RERANK_PROVIDER") or "auto").strip().lower()
    if provider in ("auto", "local", "vllm"):
        try:
            local_results = _rerank_via_local_vllm(model_cls, query, chunks_list, top_k)
            if local_results.exists():
                return local_results
        except Exception as exc:
            logger.warning("Local rerank failed; trying Cohere fallback. Error: %s", exc)
        if provider in ("local", "vllm"):
            return _fallback_rerank(model_cls, chunks_list, top_k)

    cohere = apps.get_app_config("aquillm").cohere_client  # type: ignore
    if cohere is None:
        return _fallback_rerank(model_cls, chunks_list, top_k)
    try:
        response = cohere.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=[{"content": chunk.content, "id": chunk.pk} for chunk in chunks_list],
            rank_fields=["content"],
            top_n=top_k,
            return_documents=True,
        )
        ranked_list = [result.document.id for result in response.results]
        if not ranked_list:
            return _fallback_rerank(model_cls, chunks_list, top_k)
        return _ordered_queryset_from_ids(model_cls, ranked_list)
    except Exception as exc:
        logger.warning("Cohere rerank failed; using fallback order. Error: %s", exc)
        return _fallback_rerank(model_cls, chunks_list, top_k)


__all__ = ["rerank_chunks", "rerank_document_payload"]
