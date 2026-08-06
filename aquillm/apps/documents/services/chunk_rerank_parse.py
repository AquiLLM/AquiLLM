"""Parse local vLLM / OpenAI-style rerank and score HTTP responses."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Type

from django.db.models import Case, When

if TYPE_CHECKING:
    from apps.documents.models.chunks import TextChunk


def _is_integer(value) -> bool:
    return type(value) is int


def _is_finite_number(value) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def fallback_rerank(model_cls: Type["TextChunk"], chunks, top_k: int):
    chunk_ids = [chunk.pk for chunk in list(chunks)[:top_k]]
    if not chunk_ids:
        return model_cls.objects.none()
    preserved = Case(*[When(pk=pk, then=pos) for pos, pk in enumerate(chunk_ids)])
    return model_cls.objects.filter(pk__in=chunk_ids).order_by(preserved)


def ordered_queryset_from_ids(model_cls: Type["TextChunk"], ranked_ids: list[int]):
    if not ranked_ids:
        return model_cls.objects.none()
    preserved = Case(*[When(pk=pk, then=pos) for pos, pk in enumerate(ranked_ids)])
    return model_cls.objects.filter(pk__in=ranked_ids).order_by(preserved)


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
        if not _is_integer(idx) or idx < 0 or idx >= len(chunks_list):
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
        if not _is_integer(idx):
            continue
        score = item.get("score")
        if not _is_finite_number(score):
            score = item.get("relevance_score")
        if not _is_finite_number(score):
            continue
        pairs.append((idx, float(score)))
    return pairs


def parse_single_score(body) -> float:
    if _is_finite_number(body):
        return float(body)
    if isinstance(body, dict):
        if _is_finite_number(body.get("score")):
            return float(body["score"])
        data = body.get("data")
        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict):
                for key in ("score", "relevance_score"):
                    if _is_finite_number(first.get(key)):
                        return float(first[key])
        results = body.get("results")
        if isinstance(results, list) and results:
            first = results[0]
            if isinstance(first, dict):
                for key in ("score", "relevance_score"):
                    if _is_finite_number(first.get(key)):
                        return float(first[key])
    raise ValueError(f"Unable to parse score response: {body!r}")


__all__ = [
    "fallback_rerank",
    "ordered_queryset_from_ids",
    "parse_rerank_results",
    "parse_score_results",
    "parse_single_score",
]
