"""Structured metrics for direct RAG turns."""
from __future__ import annotations

import structlog

logger = structlog.stdlib.get_logger(__name__)


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
) -> None:
    """Emit a structlog ``rag_direct_turn`` event with per-stage timing fields."""
    logger.info(
        "rag_direct_turn",
        intent_ms=round(intent_ms, 1),
        query_ms=round(query_ms, 1),
        retrieval_ms=round(retrieval_ms, 1),
        evidence_ms=round(evidence_ms, 1),
        synthesis_ms=round(synthesis_ms, 1),
        total_ms=round(total_ms, 1),
        retrieved_count=retrieved_count,
        retrieval_status=retrieval_status,
    )


__all__ = ["log_direct_rag_turn"]
