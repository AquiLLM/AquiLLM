"""Fixed, payload-free observations for exact graph readiness failures."""

from __future__ import annotations

import structlog

from lib.retrieval_redaction import RetrievalLogReason, retrieval_log_fields

logger = structlog.stdlib.get_logger(__name__)

_READINESS_REASONS = frozenset(
    {
        RetrievalLogReason.READINESS_COLLECTION_COVERAGE,
        RetrievalLogReason.READINESS_DOCUMENT_COVERAGE,
        RetrievalLogReason.READINESS_IDENTIFIER_KEY,
        RetrievalLogReason.READINESS_MEMBERSHIP_MISSING,
        RetrievalLogReason.READINESS_ARTIFACT_MISSING,
        RetrievalLogReason.READINESS_MEMBERSHIP_STALE,
        RetrievalLogReason.READINESS_ARTIFACT_BINDING,
        RetrievalLogReason.READINESS_ARTIFACT_COLLECTION,
        RetrievalLogReason.READINESS_MANIFEST,
    }
)


def record_readiness_failure(*, reason: RetrievalLogReason, elapsed_ms: float) -> None:
    """Emit one safe condition event without affecting the rejection path."""

    if type(reason) is not RetrievalLogReason or reason not in _READINESS_REASONS:
        raise TypeError("reason must be a fixed readiness reason")
    try:
        logger.info(
            "obs.rag.graph_readiness_failed",
            **retrieval_log_fields(reason=reason, count=1, elapsed_ms=elapsed_ms),
        )
    except Exception:
        # Observability must never replace the existing readiness rejection.
        pass


__all__ = ["record_readiness_failure"]
