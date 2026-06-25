"""Reranking for vector/trigram chunk candidates (local vLLM, Cohere, fallbacks)."""
from __future__ import annotations

import structlog
from os import getenv
from typing import TYPE_CHECKING, Type

from django.apps import apps

from apps.documents.services.chunk_rerank_local_vllm import rerank_via_local_vllm
from apps.documents.services.chunk_rerank_parse import fallback_rerank, ordered_queryset_from_ids
from apps.documents.services.chunk_rerank_payload import rerank_document_payload

# chunk_search and legacy imports expect this name
_fallback_rerank = fallback_rerank

if TYPE_CHECKING:
    from apps.documents.models.chunks import TextChunk

logger = structlog.stdlib.get_logger(__name__)


def rerank_chunks(model_cls: Type[TextChunk], query: str, chunks, top_k: int):
    chunks_list = list(chunks)
    provider = (getenv("APP_RERANK_PROVIDER") or "auto").strip().lower()
    if provider in ("auto", "local", "vllm"):
        try:
            local_results = rerank_via_local_vllm(model_cls, query, chunks_list, top_k)
            if local_results.exists():
                return local_results
        except Exception as exc:
            logger.warning("Local rerank failed; trying Cohere fallback. Error: %s", exc)
        if provider in ("local", "vllm"):
            return fallback_rerank(model_cls, chunks_list, top_k)

    cohere = apps.get_app_config("aquillm").cohere_client  # type: ignore
    if cohere is None:
        return fallback_rerank(model_cls, chunks_list, top_k)
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
            return fallback_rerank(model_cls, chunks_list, top_k)
        return ordered_queryset_from_ids(model_cls, ranked_list)
    except Exception as exc:
        logger.warning("Cohere rerank failed; using fallback order. Error: %s", exc)
        return fallback_rerank(model_cls, chunks_list, top_k)


__all__ = ["_fallback_rerank", "rerank_chunks", "rerank_document_payload"]
