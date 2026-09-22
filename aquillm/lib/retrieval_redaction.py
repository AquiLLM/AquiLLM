"""Closed structured fields for retrieval logs that cannot carry payloads."""

from __future__ import annotations

from enum import StrEnum
from math import isfinite

MAX_RETRIEVAL_LOG_COUNT = 1_000_000
MAX_RETRIEVAL_LOG_ELAPSED_MS = 300_000.0


class RetrievalLogReason(StrEnum):
    """Fixed retrieval outcomes safe to expose to application logs."""

    COMPLETED = "completed"
    INVALID_REQUEST = "invalid_request"
    AUTHENTICATION_FAILED = "authentication_failed"
    PAYLOAD_TOO_LARGE = "payload_too_large"
    UPSTREAM_UNAVAILABLE = "upstream_unavailable"
    PROVENANCE_MISMATCH = "provenance_mismatch"
    MIXED_ONTOLOGY = "mixed_ontology"
    EMBEDDING_UNAVAILABLE = "embedding_unavailable"
    NO_SEEDS = "no_seeds"
    AMBIGUOUS = "ambiguous"
    INTERNAL_FAILURE = "internal_failure"


def retrieval_log_fields(
    *, reason: RetrievalLogReason, count: int, elapsed_ms: float
) -> dict[str, str | int | float]:
    """Return the complete, closed shape accepted by retrieval log calls."""

    if type(reason) is not RetrievalLogReason:
        raise TypeError("reason must be an exact RetrievalLogReason")
    if type(count) is not int:
        raise TypeError("count must be an exact int")
    if not 0 <= count <= MAX_RETRIEVAL_LOG_COUNT:
        raise ValueError("count is outside the retrieval logging cap")
    if type(elapsed_ms) is not float:
        raise TypeError("elapsed_ms must be an exact float")
    if (
        not isfinite(elapsed_ms)
        or not 0.0 <= elapsed_ms <= MAX_RETRIEVAL_LOG_ELAPSED_MS
    ):
        raise ValueError("elapsed_ms is outside the retrieval logging cap")
    return {"reason": reason.value, "count": count, "elapsed_ms": elapsed_ms}


__all__ = [
    "MAX_RETRIEVAL_LOG_COUNT",
    "MAX_RETRIEVAL_LOG_ELAPSED_MS",
    "RetrievalLogReason",
    "retrieval_log_fields",
]
