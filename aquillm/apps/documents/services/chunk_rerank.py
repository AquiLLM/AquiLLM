"""Reranking for vector/trigram chunk candidates (local vLLM, Cohere, fallbacks)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from django.apps import apps

from apps.documents.services.chunk_rerank_config import rerank_provider
from apps.documents.services.chunk_rerank_local_vllm import (
    _STRICT_COMPLETE_SCORING,
    rerank_via_local_vllm,
)
from apps.documents.services.chunk_rerank_parse import (
    fallback_rerank,
    ordered_queryset_from_ids,
)
from apps.documents.services.chunk_rerank_payload import rerank_document_payload

# chunk_search and legacy imports expect this name
_fallback_rerank = fallback_rerank

if TYPE_CHECKING:
    from apps.documents.models.chunks import TextChunk

logger = structlog.stdlib.get_logger(__name__)


class StrictRerankUnavailable(RuntimeError):
    """The eval-only local reranker did not return a measured ranking."""


class _StrictEvaluationRerankCapability:
    __slots__ = ()


_STRICT_EVALUATION_RERANK = _StrictEvaluationRerankCapability()


def _strict_local_rerank_chunks(
    model_cls: type[TextChunk],
    query: str,
    chunks,
    top_k: int,
    *,
    _capability: object,
) -> tuple[TextChunk, ...]:
    """Run the shipping local adapter, but never use production fail-open output."""

    if _capability is not _STRICT_EVALUATION_RERANK:
        raise PermissionError("strict eval reranking requires its private capability")
    if rerank_provider() != "local":
        raise StrictRerankUnavailable(
            "strict eval reranking requires the exact local provider"
        )
    chunks_list = list(chunks)
    if not chunks_list:
        raise StrictRerankUnavailable("strict eval reranking candidate pool is empty")
    candidate_ids = tuple(getattr(row, "pk", None) for row in chunks_list)
    if any(
        type(identifier) is not int or identifier <= 0 for identifier in candidate_ids
    ):
        raise StrictRerankUnavailable(
            "strict eval reranking requires positive candidate IDs"
        )
    if len(set(candidate_ids)) != len(candidate_ids):
        raise StrictRerankUnavailable(
            "strict eval reranking requires a unique candidate pool"
        )
    ranked = tuple(
        rerank_via_local_vllm(
            model_cls,
            query,
            chunks_list,
            top_k,
            _complete_scoring_capability=_STRICT_COMPLETE_SCORING,
        )
    )
    if not ranked:
        raise StrictRerankUnavailable("strict local reranker returned an empty result")
    expected_count = min(top_k, len(chunks_list))
    if len(ranked) != expected_count:
        raise StrictRerankUnavailable(
            "strict local reranker did not return a complete ranking"
        )
    ranked_ids = tuple(getattr(row, "pk", None) for row in ranked)
    if len(set(ranked_ids)) != len(ranked_ids):
        raise StrictRerankUnavailable(
            "strict local reranker did not return unique rows"
        )
    if any(identifier not in set(candidate_ids) for identifier in ranked_ids):
        raise StrictRerankUnavailable(
            "strict local reranker returned a row outside the candidate pool"
        )
    return ranked


def rerank_chunks(model_cls: type[TextChunk], query: str, chunks, top_k: int):
    chunks_list = list(chunks)
    provider = rerank_provider()
    if provider in ("auto", "local", "vllm"):
        try:
            local_results = rerank_via_local_vllm(model_cls, query, chunks_list, top_k)
            if local_results.exists():
                return local_results
        except Exception as exc:
            logger.warning(
                "obs.rag.local_rerank_failed",
                error=str(exc),
                error_type=type(exc).__name__,
            )
        if provider in ("local", "vllm"):
            return fallback_rerank(model_cls, chunks_list, top_k)

    cohere = apps.get_app_config("aquillm").cohere_client  # type: ignore
    if cohere is None:
        return fallback_rerank(model_cls, chunks_list, top_k)
    try:
        response = cohere.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=[
                {"content": chunk.content, "id": chunk.pk} for chunk in chunks_list
            ],
            rank_fields=["content"],
            top_n=top_k,
            return_documents=True,
        )
        ranked_list = [result.document.id for result in response.results]
        if not ranked_list:
            return fallback_rerank(model_cls, chunks_list, top_k)
        return ordered_queryset_from_ids(model_cls, ranked_list)
    except Exception as exc:
        logger.warning(
            "obs.rag.cohere_rerank_failed",
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return fallback_rerank(model_cls, chunks_list, top_k)


__all__ = ["_fallback_rerank", "rerank_chunks", "rerank_document_payload"]
