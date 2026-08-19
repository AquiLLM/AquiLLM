"""Structured metrics for direct RAG turns."""

from __future__ import annotations

import re
from math import isfinite

import structlog

logger = structlog.stdlib.get_logger(__name__)

_GRAPH_STATUSES = frozenset({"miss", "hit", "timeout", "error"})
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_MAX_GRAPH_METRIC_MS = 60_000.0
_MAX_GRAPH_SEEDS = 64
_MAX_GRAPH_CANDIDATES = 20


def _safe_graph_ms(value: object) -> float | None:
    if type(value) not in (int, float):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    if not isfinite(number) or not 0.0 <= number <= _MAX_GRAPH_METRIC_MS:
        return None
    return round(number, 1)


def _safe_graph_count(value: object, *, maximum: int) -> int | None:
    if type(value) is not int or not 0 <= value <= maximum:
        return None
    return value


def _safe_graph_status(value: object) -> str | None:
    if type(value) is not str or value not in _GRAPH_STATUSES:
        return None
    return value


def _safe_graph_signature(value: object) -> str | None:
    if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
        return None
    return value


def log_direct_rag_turn(
    *,
    intent_ms: float,
    query_ms: float,
    retrieval_ms: float,
    evidence_ms: float,
    synthesis_ms: float,
    total_ms: float,
    retrieved_count: int,
    retrieval_status: str,
    graph_ms: float | None = None,
    graph_seed_count: int | None = None,
    graph_candidate_count: int | None = None,
    graph_status: str | None = None,
    graph_algorithm_signature: str | None = None,
    graph_version_signature: str | None = None,
) -> None:
    """Emit a structlog ``rag_direct_turn`` event with per-stage timing fields."""
    fields = {
        "intent_ms": round(intent_ms, 1),
        "query_ms": round(query_ms, 1),
        "retrieval_ms": round(retrieval_ms, 1),
        "evidence_ms": round(evidence_ms, 1),
        "synthesis_ms": round(synthesis_ms, 1),
        "total_ms": round(total_ms, 1),
        "retrieved_count": retrieved_count,
        "retrieval_status": retrieval_status,
    }
    optional_graph_fields = {
        "graph_ms": _safe_graph_ms(graph_ms),
        "graph_seed_count": _safe_graph_count(
            graph_seed_count,
            maximum=_MAX_GRAPH_SEEDS,
        ),
        "graph_candidate_count": _safe_graph_count(
            graph_candidate_count,
            maximum=_MAX_GRAPH_CANDIDATES,
        ),
        "graph_status": _safe_graph_status(graph_status),
        "graph_algorithm_signature": _safe_graph_signature(graph_algorithm_signature),
        "graph_version_signature": _safe_graph_signature(graph_version_signature),
    }
    fields.update(
        {
            key: value
            for key, value in optional_graph_fields.items()
            if value is not None
        }
    )
    logger.info("rag_direct_turn", **fields)


__all__ = ["log_direct_rag_turn"]
