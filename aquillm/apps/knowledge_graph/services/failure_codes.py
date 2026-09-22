"""Safe failure categories persisted by build orchestration."""

from __future__ import annotations

import uuid
from enum import StrEnum

DOCUMENT_CAPACITY_FAILURE_CODES = frozenset(
    {
        "extraction_chunk_limit",
        "extraction_character_limit",
        "extraction_entity_limit",
        "extraction_relation_limit",
        "extraction_observation_limit",
    }
)


class BuildLeaseLostError(RuntimeError):
    """The caller no longer owns the durable attempt generation."""


class BuildInProgressError(RuntimeError):
    """Another live worker currently owns this exact build identity."""


class StaleBuildError(RuntimeError):
    """The immutable requested source no longer matches live source state."""


class CorruptBuildError(RuntimeError):
    """Persisted rows cannot be tied to a complete commit marker."""


class RebuildPublicationError(RuntimeError):
    """Durable request work remains resumable after broker publication failed."""

    def __init__(self, request_id: uuid.UUID, error_code: str) -> None:
        self.request_id = request_id
        self.error_code = error_code
        super().__init__(
            f"rebuild request {request_id} publication failed: {error_code}"
        )


class CommitMarkerState(StrEnum):
    """Durable stage commit state derived from both marker and persisted rows."""

    ABSENT = "absent"
    VALID = "valid"
    CORRUPT = "corrupt"
