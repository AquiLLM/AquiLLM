"""Reranking for vector/trigram chunk candidates (local vLLM, Cohere, fallbacks)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from django.apps import apps

from apps.documents.services.chunk_rerank_config import rerank_provider
from apps.documents.services.chunk_rerank_local_scored import (
    rerank_via_local_vllm_scored,
)
from apps.documents.services.chunk_rerank_local_vllm import (
    _STRICT_COMPLETE_SCORING,
    rerank_via_local_vllm,
)
from apps.documents.services.chunk_rerank_parse import (
    fallback_rerank,
    ordered_queryset_from_ids,
)
from apps.documents.services.chunk_rerank_payload import rerank_document_payload
from apps.documents.services.chunk_rerank_results import (
    PassageScore,
    RerankScoreSet,
    ScoredRerankResult,
    fingerprint_pair,
    fingerprint_pool,
    fingerprint_text,
)
from lib.retrieval_redaction import RetrievalLogReason, retrieval_log_fields

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


def rerank_chunks(
    model_cls: type[TextChunk], query: str, chunks, top_k: int, *, turn_budget=None
):
    chunks_list = list(chunks)
    provider = rerank_provider()
    if provider in ("auto", "local", "vllm"):
        try:
            local_results = rerank_via_local_vllm(
                model_cls, query, chunks_list, top_k, turn_budget=turn_budget
            )
            if local_results.exists():
                return local_results
        except Exception:
            logger.warning(
                "obs.rag.local_rerank_failed",
                **retrieval_log_fields(
                    reason=RetrievalLogReason.UPSTREAM_UNAVAILABLE,
                    count=0,
                    elapsed_ms=0.0,
                ),
            )
        return fallback_rerank(model_cls, chunks_list, top_k)

    result = rerank_chunks_scored(
        model_cls, query, chunks_list, top_k, turn_budget=turn_budget
    )
    return ordered_queryset_from_ids(model_cls, result.ranked_ids)


def rerank_chunks_scored(
    model_cls: type[TextChunk],
    query: str,
    chunks,
    top_k: int,
    *,
    turn_budget=None,
    window_scorer=None,
) -> ScoredRerankResult:
    """Capture provider scores from the ranking call, without replaying inference."""
    chunks_list = list(chunks)
    from .rerank_request_lifetime import rerank_state_for

    state = rerank_state_for(turn_budget)
    turn_budget = state.budget
    from .chunk_rerank_window_acquisition import dispatch_windowed

    candidate_ids = tuple(chunk.pk for chunk in chunks_list)
    pool = fingerprint_pool(
        tuple(
            (
                chunk.pk,
                fingerprint_text(chunk.content),
                fingerprint_pair(query, chunk.content),
            )
            for chunk in chunks_list
        )
    )
    query_fp = fingerprint_text(query)

    def unavailable(ranked_ids=()):
        return ScoredRerankResult(
            tuple(ranked_ids)[:top_k],
            RerankScoreSet(
                "v2",
                query_fp,
                "",
                pool,
                "rank_only",
                "unavailable",
                candidate_ids,
                (),
            ),
        )

    provider = rerank_provider()
    if not state.can_admit():
        return unavailable(candidate_ids[:top_k])
    windowed = dispatch_windowed(
        query, chunks_list, top_k, turn_budget, window_scorer, shadow=True
    )
    if windowed is not None:
        return windowed
    if provider in ("auto", "local", "vllm"):
        try:
            local = rerank_via_local_vllm_scored(
                model_cls,
                query,
                chunks_list,
                top_k,
                turn_budget=turn_budget,
                rerank_state=state,
            )
            if local.ranked_ids:
                return local
        except Exception:
            logger.warning(
                "obs.rag.local_rerank_failed",
                **retrieval_log_fields(
                    reason=RetrievalLogReason.UPSTREAM_UNAVAILABLE,
                    count=0,
                    elapsed_ms=0.0,
                ),
            )
        return unavailable(candidate_ids[:top_k])

    cohere = apps.get_app_config("aquillm").cohere_client  # type: ignore
    if cohere is None:
        return unavailable(candidate_ids[:top_k])
    try:
        from math import isfinite

        from .chunk_rerank_config import rerank_timeout_seconds

        width = min(6, turn_budget.limits.in_flight_pairs)
        if width <= 0:
            return unavailable(candidate_ids[:top_k])
        all_results = []
        for start in range(0, len(chunks_list), width):
            batch = chunks_list[start : start + width]

            def send(remaining):
                response = cohere.rerank(
                    model="rerank-english-v3.0",
                    query=query,
                    documents=[
                        {"content": chunk.content, "id": chunk.pk} for chunk in batch
                    ],
                    rank_fields=["content"],
                    top_n=len(batch),
                    return_documents=True,
                    request_options={
                        "timeout_in_seconds": min(rerank_timeout_seconds(), remaining),
                        "max_retries": 0,
                    },
                )
                results = tuple(response.results)
                if len(results) != len(batch) or {r.document.id for r in results} != {
                    c.pk for c in batch
                }:
                    return None
                if any(
                    type(getattr(r, "relevance_score", None)) not in (int, float)
                    or not isfinite(float(r.relevance_score))
                    for r in results
                ):
                    return None
                return results

            results = state.run(len(batch), send, timeout=rerank_timeout_seconds())
            if results is None:
                return unavailable(candidate_ids[:top_k])
            all_results.extend(results)
        if not state.can_admit():
            return unavailable(candidate_ids[:top_k])
        ranked_ids = tuple(result.document.id for result in all_results)
        by_id = {chunk.pk: chunk for chunk in chunks_list}
        if (
            len(ranked_ids) != len(candidate_ids)
            or len(set(ranked_ids)) != len(candidate_ids)
            or set(ranked_ids) != set(candidate_ids)
        ):
            return unavailable(ranked_ids)
        scores_by_id = {}
        for result in all_results:
            value = getattr(result, "relevance_score", None)
            if type(value) not in (int, float):
                return unavailable(ranked_ids)
            chunk = by_id[result.document.id]
            scores_by_id[chunk.pk] = PassageScore(
                chunk.pk,
                chunk.doc_id,
                chunk.chunk_number,
                fingerprint_text(chunk.content),
                fingerprint_pair(query, chunk.content),
                float(value),
            )

        if not all(isfinite(score.value) for score in scores_by_id.values()):
            return unavailable(ranked_ids)
        scores = tuple(scores_by_id[pk] for pk in candidate_ids)
        stable_ranked = tuple(
            sorted(
                candidate_ids,
                key=lambda pk: (-scores_by_id[pk].value, candidate_ids.index(pk)),
            )
        )
        return ScoredRerankResult(
            stable_ranked[:top_k],
            RerankScoreSet(
                "v2",
                query_fp,
                fingerprint_text("cohere:rerank-english-v3.0"),
                pool,
                "listwise",
                "complete",
                candidate_ids,
                scores,
            ),
        )
    except Exception:
        logger.warning(
            "obs.rag.cohere_rerank_failed",
            **retrieval_log_fields(
                reason=RetrievalLogReason.UPSTREAM_UNAVAILABLE, count=0, elapsed_ms=0.0
            ),
        )
        return unavailable(candidate_ids[:top_k])


__all__ = [
    "_fallback_rerank",
    "rerank_chunks",
    "rerank_chunks_scored",
    "rerank_document_payload",
]
