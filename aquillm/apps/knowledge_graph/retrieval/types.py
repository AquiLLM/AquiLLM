"""Immutable, ORM-neutral contracts for graph retrieval expansion."""

from dataclasses import dataclass
from math import isfinite
from sys import float_info
from uuid import UUID


_GRAPH_EXPANSION_STATUSES = frozenset({"disabled", "miss", "hit", "timeout", "error"})


def _require_nonempty_string(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a nonempty string")


def _require_positive_chunk_ids(value: object, field_name: str, *, nonempty: bool) -> None:
    if not isinstance(value, tuple):
        raise ValueError(f"{field_name} must be a tuple")
    if nonempty and not value:
        raise ValueError(f"{field_name} must not be empty")
    if any(type(chunk_id) is not int or chunk_id <= 0 for chunk_id in value):
        raise ValueError(f"{field_name} must contain positive integer chunk IDs")


def _require_uuid_tuple(value: object, field_name: str) -> None:
    if not isinstance(value, tuple):
        raise ValueError(f"{field_name} must be a tuple")
    if not value:
        raise ValueError(f"{field_name} must not be empty")
    if not all(isinstance(identifier, UUID) for identifier in value):
        raise ValueError(f"{field_name} must contain UUID values")


@dataclass(frozen=True, slots=True)
class GraphExpansionRequest:
    """A scoped request to expand graph retrieval from retrieved chunk seeds."""

    query: str
    seed_chunk_ids: tuple[int, ...]
    allowed_doc_ids: tuple[UUID, ...]
    allowed_collection_ids: tuple[UUID, ...]

    def __post_init__(self) -> None:
        _require_nonempty_string(self.query, "query")
        _require_positive_chunk_ids(self.seed_chunk_ids, "seed_chunk_ids", nonempty=True)
        _require_uuid_tuple(self.allowed_doc_ids, "allowed_doc_ids")
        _require_uuid_tuple(self.allowed_collection_ids, "allowed_collection_ids")


@dataclass(frozen=True, slots=True)
class GraphExpansionDiagnostics:
    """Safe operational metadata for a graph expansion attempt."""

    status: str
    candidate_count: int = 0
    elapsed_ms: float | None = None
    version_signature: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, str) or self.status not in _GRAPH_EXPANSION_STATUSES:
            raise ValueError("status must be disabled, miss, hit, timeout, or error")
        if type(self.candidate_count) is not int or self.candidate_count < 0:
            raise ValueError("candidate_count must be a nonnegative integer")
        if self.elapsed_ms is not None:
            if type(self.elapsed_ms) not in (int, float):
                raise ValueError("elapsed_ms must be a finite nonnegative number")
            if type(self.elapsed_ms) is float and not isfinite(self.elapsed_ms):
                raise ValueError("elapsed_ms must be a finite nonnegative number")
            if self.elapsed_ms < 0 or (
                type(self.elapsed_ms) is int and self.elapsed_ms > float_info.max
            ):
                raise ValueError("elapsed_ms must be a finite nonnegative number")
        if self.version_signature is not None:
            _require_nonempty_string(self.version_signature, "version_signature")


@dataclass(frozen=True, slots=True)
class GraphExpansionResult:
    """Graph-provided chunk IDs plus safe diagnostics for the expansion attempt."""

    chunk_ids: tuple[int, ...]
    diagnostics: GraphExpansionDiagnostics

    def __post_init__(self) -> None:
        _require_positive_chunk_ids(self.chunk_ids, "chunk_ids", nonempty=False)
        if not isinstance(self.diagnostics, GraphExpansionDiagnostics):
            raise ValueError("diagnostics must be a GraphExpansionDiagnostics value")
