"""Structured metrics for direct RAG turns."""
from __future__ import annotations

import structlog

from lib.llm.providers.request_observability import safe_correlation_id

logger = structlog.stdlib.get_logger(__name__)


def log_direct_rag_turn(
    *,
    correlation_id: str | None = None,
    intent_ms: float,
    query_ms: float,
    retrieval_ms: float,
    evidence_ms: float,
    synthesis_ms: float,
    persistence_ms: float = 0.0,
    total_ms: float,
    retrieved_count: int,
    retained_count: int | None = None,
    retrieval_status: str,
) -> None:
    """Emit a structlog ``rag_direct_turn`` event with per-stage timing fields."""
    fields = {
        "correlation_id": safe_correlation_id(correlation_id),
        "intent_ms": round(intent_ms, 1),
        "query_ms": round(query_ms, 1),
        "retrieval_ms": round(retrieval_ms, 1),
        "evidence_ms": round(evidence_ms, 1),
        "synthesis_ms": round(synthesis_ms, 1),
        "persistence_ms": round(persistence_ms, 1),
        "total_ms": round(total_ms, 1),
        "retrieved_count": retrieved_count,
        "retained_count": (
            retrieved_count if retained_count is None else retained_count
        ),
        "retrieval_status": retrieval_status,
    }
    logger.info("rag_direct_turn", **fields)


__all__ = ["log_direct_rag_turn"]
