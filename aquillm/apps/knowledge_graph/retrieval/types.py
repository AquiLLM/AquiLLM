"""Immutable, ORM-neutral contracts for graph retrieval expansion."""

from __future__ import annotations

import re
from dataclasses import InitVar, dataclass
from math import isfinite
from uuid import UUID

from .ppr import PPRAlgorithmConfig

_GRAPH_EXPANSION_STATUSES = frozenset({"miss", "hit", "timeout", "error"})
_HASH_PATTERN = re.compile(r"[0-9a-f]{64}")
_DATABASE_ID_MAX = 2**63 - 1
_LIMITS = PPRAlgorithmConfig(canonical_resolver_version="retrieval-contract-v1")


def _positive_database_int(value: object, field_name: str) -> int:
    if type(value) is not int or not 1 <= value <= _DATABASE_ID_MAX:
        raise ValueError(f"{field_name} must be a positive database integer")
    return value


def _positive_float(value: object, field_name: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{field_name} must be a finite positive number")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError(f"{field_name} must be a finite positive number") from error
    if not isfinite(number) or number <= 0.0:
        raise ValueError(f"{field_name} must be a finite positive number")
    return number


def _require_bounded_tuple(
    value: object,
    field_name: str,
    *,
    maximum: int,
    nonempty: bool,
) -> tuple[object, ...]:
    if type(value) is not tuple:
        raise ValueError(f"{field_name} must be an exact tuple")
    if len(value) > maximum:
        raise ValueError(f"{field_name} exceeds the hard cap of {maximum}")
    if nonempty and not value:
        raise ValueError(f"{field_name} must not be empty")
    return value


def _require_unique(values: tuple[object, ...], field_name: str) -> None:
    try:
        unique_count = len(set(values))
    except TypeError as error:
        raise ValueError(f"{field_name} values must be exact and hashable") from error
    if unique_count != len(values):
        raise ValueError(f"{field_name} values must be unique")


def _optional_sha256(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    if type(value) is not str or _HASH_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be an exact lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class GraphExpansionConfig:
    """Narrow immutable acquisition settings shared with baseline RAG fusion."""

    rrf_k: int
    max_seeds: int
    max_scope_documents: int
    max_scope_collections: int
    max_candidates: int
    algorithm_signature: str

    def __post_init__(self) -> None:
        limits = {
            "rrf_k": 1_000,
            "max_seeds": _LIMITS.max_seeds,
            "max_scope_documents": _LIMITS.max_scope_documents,
            "max_scope_collections": _LIMITS.max_scope_collections,
            "max_candidates": _LIMITS.max_candidates,
        }
        for field_name, maximum in limits.items():
            value = getattr(self, field_name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(
                    f"{field_name} must be an exact integer in [1, {maximum}]"
                )
        if self.max_scope_collections > self.max_scope_documents:
            raise ValueError(
                "max_scope_collections must not exceed max_scope_documents"
            )
        signature = _optional_sha256(
            self.algorithm_signature,
            "algorithm_signature",
        )
        if signature is None:
            raise ValueError("algorithm_signature must be an exact SHA-256 digest")
        object.__setattr__(self, "algorithm_signature", signature)


@dataclass(frozen=True, slots=True)
class GraphExpansionSeed:
    """One ranked baseline-RAG chunk contributing personalized restart mass."""

    chunk_id: int
    rank: int
    restart_weight: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "chunk_id", _positive_database_int(self.chunk_id, "chunk_id")
        )
        object.__setattr__(self, "rank", _positive_database_int(self.rank, "rank"))
        object.__setattr__(
            self,
            "restart_weight",
            _positive_float(self.restart_weight, "restart_weight"),
        )


@dataclass(frozen=True, slots=True)
class GraphExpansionRequest:
    """A frozen permission scope and weighted seed snapshot for graph expansion."""

    seeds: tuple[GraphExpansionSeed, ...]
    allowed_doc_ids: tuple[UUID, ...]
    allowed_collection_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        raw_seeds = _require_bounded_tuple(
            self.seeds,
            "seeds",
            maximum=_LIMITS.max_seeds,
            nonempty=True,
        )
        if any(type(seed) is not GraphExpansionSeed for seed in raw_seeds):
            raise ValueError("seeds must contain exact GraphExpansionSeed values")
        seeds = tuple(raw_seeds)
        chunk_ids = tuple(seed.chunk_id for seed in seeds)
        ranks = tuple(seed.rank for seed in seeds)
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("seed chunk_id values must be unique")
        if len(set(ranks)) != len(ranks):
            raise ValueError("seed rank values must be unique")
        if ranks != tuple(sorted(ranks)):
            raise ValueError("seeds must be ordered by increasing rank")

        raw_doc_ids = _require_bounded_tuple(
            self.allowed_doc_ids,
            "allowed_doc_ids",
            maximum=_LIMITS.max_scope_documents,
            nonempty=True,
        )
        if any(type(identifier) is not UUID for identifier in raw_doc_ids):
            raise ValueError("allowed_doc_ids must contain exact UUID values")
        _require_unique(raw_doc_ids, "allowed_doc_ids")
        doc_ids = tuple(raw_doc_ids)
        if doc_ids != tuple(sorted(doc_ids, key=lambda identifier: identifier.int)):
            raise ValueError("allowed_doc_ids must be canonically sorted")

        raw_collection_ids = _require_bounded_tuple(
            self.allowed_collection_ids,
            "allowed_collection_ids",
            maximum=_LIMITS.max_scope_collections,
            nonempty=True,
        )
        collection_ids = tuple(
            _positive_database_int(identifier, "allowed_collection_ids")
            for identifier in raw_collection_ids
        )
        _require_unique(collection_ids, "allowed_collection_ids")
        if collection_ids != tuple(sorted(collection_ids)):
            raise ValueError("allowed_collection_ids must be canonically sorted")
        if len(collection_ids) > len(doc_ids):
            raise ValueError(
                "allowed_collection_ids cannot outnumber authorized documents"
            )

        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "allowed_doc_ids", doc_ids)
        object.__setattr__(self, "allowed_collection_ids", collection_ids)


@dataclass(frozen=True, slots=True)
class GraphExpansionDiagnostics:
    """Non-enumerating operational metadata for one expansion attempt."""

    status: str
    seed_count: int = 0
    candidate_count: int = 0
    elapsed_ms: float | None = None
    algorithm_signature: str | None = None
    graph_version_signature: str | None = None

    def __post_init__(self) -> None:
        if type(self.status) is not str or self.status not in _GRAPH_EXPANSION_STATUSES:
            raise ValueError("status must be miss, hit, timeout, or error")
        if (
            type(self.seed_count) is not int
            or not 0 <= self.seed_count <= _LIMITS.max_seeds
        ):
            raise ValueError("seed_count must be a bounded nonnegative exact integer")
        if (
            type(self.candidate_count) is not int
            or not 0 <= self.candidate_count <= _LIMITS.max_candidates
        ):
            raise ValueError(
                "candidate_count must be a bounded nonnegative exact integer"
            )
        if self.elapsed_ms is not None:
            if type(self.elapsed_ms) not in (int, float):
                raise ValueError("elapsed_ms must be a finite nonnegative number")
            try:
                elapsed_ms = float(self.elapsed_ms)
            except OverflowError as error:
                raise ValueError(
                    "elapsed_ms must be a finite nonnegative number"
                ) from error
            if not isfinite(elapsed_ms) or elapsed_ms < 0.0:
                raise ValueError("elapsed_ms must be a finite nonnegative number")
            object.__setattr__(self, "elapsed_ms", elapsed_ms)
        object.__setattr__(
            self,
            "algorithm_signature",
            _optional_sha256(self.algorithm_signature, "algorithm_signature"),
        )
        object.__setattr__(
            self,
            "graph_version_signature",
            _optional_sha256(self.graph_version_signature, "graph_version_signature"),
        )


@dataclass(frozen=True, slots=True)
class GraphExpansionResult:
    """Ordered novel chunks plus privacy-safe operational diagnostics."""

    chunk_ids: tuple[int, ...]
    diagnostics: GraphExpansionDiagnostics
    seed_chunk_ids: InitVar[tuple[int, ...]]

    def __post_init__(self, seed_chunk_ids: tuple[int, ...]) -> None:
        raw_chunk_ids = _require_bounded_tuple(
            self.chunk_ids,
            "chunk_ids",
            maximum=_LIMITS.max_candidates,
            nonempty=False,
        )
        chunk_ids = tuple(
            _positive_database_int(identifier, "chunk_ids")
            for identifier in raw_chunk_ids
        )
        _require_unique(chunk_ids, "chunk_ids")

        raw_seed_ids = _require_bounded_tuple(
            seed_chunk_ids,
            "seed_chunk_ids",
            maximum=_LIMITS.max_seeds,
            nonempty=True,
        )
        seed_ids = tuple(
            _positive_database_int(identifier, "seed_chunk_ids")
            for identifier in raw_seed_ids
        )
        _require_unique(seed_ids, "seed_chunk_ids")
        if set(chunk_ids).intersection(seed_ids):
            raise ValueError("chunk_ids must contain only novel non-seed chunks")

        if type(self.diagnostics) is not GraphExpansionDiagnostics:
            raise ValueError(
                "diagnostics must be an exact GraphExpansionDiagnostics value"
            )
        if self.diagnostics.seed_count != len(seed_ids):
            raise ValueError("diagnostics seed_count must match the seed snapshot")
        if self.diagnostics.candidate_count != len(chunk_ids):
            raise ValueError("diagnostics candidate_count must match chunk_ids")
        if bool(chunk_ids) != (self.diagnostics.status == "hit"):
            raise ValueError("diagnostics status must be hit exactly when chunks exist")

        object.__setattr__(self, "chunk_ids", chunk_ids)


__all__ = [
    "GraphExpansionConfig",
    "GraphExpansionDiagnostics",
    "GraphExpansionRequest",
    "GraphExpansionResult",
    "GraphExpansionSeed",
]
