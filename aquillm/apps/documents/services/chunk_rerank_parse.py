"""Parse local vLLM / OpenAI-style rerank and score HTTP responses."""

from __future__ import annotations

from math import isfinite
from typing import TYPE_CHECKING

from django.db.models import Case, When

if TYPE_CHECKING:
    from apps.documents.models.chunks import TextChunk


def _finite_score(value) -> float | None:
    if type(value) not in (int, float):
        return None
    result = float(value)
    return result if isfinite(result) else None


def fallback_rerank(model_cls: type[TextChunk], chunks, top_k: int):
    chunk_ids = [chunk.pk for chunk in list(chunks)[:top_k]]
    return ordered_queryset_from_ids(model_cls, chunk_ids, status="unavailable")


def ordered_queryset_from_ids(
    model_cls: type[TextChunk], ranked_ids: list[int], *, status="unknown"
):
    preserved = Case(*[When(pk=pk, then=pos) for pos, pk in enumerate(ranked_ids)])
    rows = (
        model_cls.objects.filter(pk__in=ranked_ids).order_by(preserved)
        if ranked_ids
        else model_cls.objects.none()
    )
    return mark_rerank_status(rows, status)


def mark_rerank_status(rows, status):
    normalized = (
        status
        if type(status) is str and status in ("complete", "unavailable")
        else "unknown"
    )
    if hasattr(rows, "__dict__"):
        rows._rag_rerank_status = normalized
    return rows


def queryset_from_scored_result(
    model_cls, result, *, orderer=ordered_queryset_from_ids
):
    return mark_rerank_status(
        orderer(model_cls, result.ranked_ids), result.score_set.status
    )


def parse_rerank_results(body, chunks_list) -> list[int]:
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


def parse_score_results(body) -> list[tuple[int, float]]:
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
        score = _finite_score(item.get("score"))
        if score is None:
            score = _finite_score(item.get("relevance_score"))
        if score is None:
            continue
        pairs.append((idx, score))
    return pairs


def parse_single_score(body) -> float:
    direct = _finite_score(body)
    if direct is not None:
        return direct
    if isinstance(body, dict):
        score = _finite_score(body.get("score"))
        if score is not None:
            return score
        data = body.get("data")
        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict):
                for key in ("score", "relevance_score"):
                    score = _finite_score(first.get(key))
                    if score is not None:
                        return score
        results = body.get("results")
        if isinstance(results, list) and results:
            first = results[0]
            if isinstance(first, dict):
                for key in ("score", "relevance_score"):
                    score = _finite_score(first.get(key))
                    if score is not None:
                        return score
    raise ValueError(f"Unable to parse score response: {body!r}")


__all__ = [
    "fallback_rerank",
    "ordered_queryset_from_ids",
    "parse_rerank_results",
    "parse_score_results",
    "parse_single_score",
]
