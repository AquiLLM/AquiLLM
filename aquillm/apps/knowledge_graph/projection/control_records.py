from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from .serialization import _count, _key, _token, _uuid

_MAX_PRIVATE_PK = 2**63 - 1
_MAX_ATTEMPTS = 32767


class ProjectionLifecycleState(StrEnum):
    PENDING = "pending"
    BUILDING = "building"
    READY = "ready"
    FAILED = "failed"
    SUPERSEDED = "superseded"


class ProjectionFailureCode(StrEnum):
    SOURCE_CHANGED = "source_changed"
    LEASE_LOST = "lease_lost"
    GRAPH_UNAVAILABLE = "graph_unavailable"
    WRITE_FAILED = "write_failed"
    VALIDATION_FAILED = "validation_failed"
    CHECKSUM_MISMATCH = "checksum_mismatch"
    TIMEOUT = "timeout"
    INTERNAL_ERROR = "internal_error"


@dataclass(frozen=True, slots=True)
class ProjectionLeaseV1:
    projection_id: str
    owner: str
    expires_at: datetime
    attempt_count: int

    def __post_init__(self) -> None:
        _uuid(self.projection_id, "projection_id")
        _token(self.owner, "owner", maximum=128)
        if type(self.expires_at) is not datetime or self.expires_at.tzinfo is not UTC:
            raise ValueError("expires_at must be an exact UTC datetime")
        _count(self.attempt_count, "attempt_count", maximum=_MAX_ATTEMPTS)


@dataclass(frozen=True, slots=True)
class ProjectionFailureStateV1:
    state: ProjectionLifecycleState
    failure_code: ProjectionFailureCode
    attempt_count: int

    def __post_init__(self) -> None:
        if self.state is not ProjectionLifecycleState.FAILED:
            raise ValueError("failure state must be failed")
        if type(self.failure_code) is not ProjectionFailureCode:
            raise TypeError("failure_code must be ProjectionFailureCode")
        _count(self.attempt_count, "attempt_count", maximum=_MAX_ATTEMPTS)


@dataclass(frozen=True, slots=True)
class PrivateProjectionChunkReferenceV1:
    projection_chunk_key: str
    integer_chunk_pk: int
    document_uuid: str
    chunk_number: int

    def __post_init__(self) -> None:
        _key(self.projection_chunk_key, "projection_chunk_key")
        _count(
            self.integer_chunk_pk,
            "integer_chunk_pk",
            minimum=1,
            maximum=_MAX_PRIVATE_PK,
        )
        _uuid(self.document_uuid, "document_uuid")
        _count(self.chunk_number, "chunk_number")
