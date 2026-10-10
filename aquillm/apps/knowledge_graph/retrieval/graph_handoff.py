"""Private bounded graph handoff; no client key admission or source text."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from ._graph_v2_validation import (
    canonical_bytes,
    checksum,
    count,
    digest,
    ordered_strings,
    reference_keys,
    score,
)

GraphCoverageReason = Literal[
    "not_ready",
    "incompatible_generation",
    "frontier_budget",
    "evidence_budget",
    "deadline",
]
_REASONS = frozenset(
    {
        "not_ready",
        "incompatible_generation",
        "frontier_budget",
        "evidence_budget",
        "deadline",
    }
)


@dataclass(frozen=True, slots=True)
class GraphCoverageV2:
    selected_documents: int
    ready_documents: int
    status: Literal["complete", "partial", "unavailable"]
    reasons: tuple[GraphCoverageReason, ...]
    exhaustive: bool

    def __post_init__(self):
        count(self.selected_documents, "selected_documents", 10000)
        count(self.ready_documents, "ready_documents", self.selected_documents)
        if type(self.status) is not str or self.status not in {
            "complete",
            "partial",
            "unavailable",
        }:
            raise ValueError("unsupported graph coverage status")

        def reason(value):
            if type(value) is not str or value not in _REASONS:
                raise ValueError("unsupported graph coverage reason")

        ordered_strings(self.reasons, "reasons", reason, len(_REASONS))
        if type(self.exhaustive) is not bool or self.exhaustive is not False:
            raise ValueError("bounded graph coverage must be non-exhaustive")
        if self.status == "complete" and (
            self.ready_documents != self.selected_documents or self.reasons
        ):
            raise ValueError(
                "complete coverage cannot carry missing scope or partiality"
            )
        if self.status == "partial" and (not self.ready_documents or not self.reasons):
            raise ValueError("partial coverage requires ready scope and a reason")
        if self.status == "unavailable" and self.ready_documents:
            raise ValueError("unavailable coverage cannot carry ready documents")


@dataclass(frozen=True, slots=True)
class GraphCandidateRefV2:
    chunk_key: str
    graph_generation_key: str
    representation_generation_key: UUID | None
    authority_group_checksum: str
    ready_bundle_checksum: str
    evidence_provenance_keys: tuple[str, ...]
    relation_types: tuple[str, ...]
    score: float

    def __post_init__(self):
        reference_keys(self)
        checksum(self.ready_bundle_checksum, "ready_bundle_checksum")
        score(self.score, "score")


@dataclass(frozen=True, slots=True)
class GraphHandoffV2:
    candidates: tuple[GraphCandidateRefV2, ...]
    coverage: GraphCoverageV2
    algorithm_signature: str

    def __post_init__(self):
        if type(self.coverage) is not GraphCoverageV2:
            raise TypeError("coverage must be exact GraphCoverageV2")
        checksum(self.algorithm_signature, "algorithm_signature")
        _validate_candidates(self.candidates, self.coverage, maximum=20)


def _validate_candidates(candidates, coverage, *, maximum):
    if type(candidates) is not tuple or any(
        type(c) is not GraphCandidateRefV2 for c in candidates
    ):
        raise TypeError(
            "candidates must contain exact private GraphCandidateRefV2 values"
        )
    count(len(candidates), "candidate count", maximum)
    keys = tuple(c.chunk_key for c in candidates)
    if len(set(keys)) != len(keys):
        raise ValueError("candidate chunk keys must be unique")
    if candidates != tuple(sorted(candidates, key=lambda c: (-c.score, c.chunk_key))):
        raise ValueError("candidates must be in descending score and stable key order")
    if candidates and (
        coverage.status == "unavailable" or not coverage.ready_documents
    ):
        raise ValueError("unavailable graph cannot carry candidates")


def validate_graph_handoff_v2(value: GraphHandoffV2, *, maximum: int = 20) -> None:
    """Only a captured server DTO is accepted; mappings are never deserialized."""
    if type(value) is not GraphHandoffV2:
        raise TypeError("handoff must be an exact private GraphHandoffV2")
    count(maximum, "maximum", 20)
    if maximum < 1:
        raise ValueError("maximum must be positive")
    _validate_candidates(value.candidates, value.coverage, maximum=maximum)


def canonical_graph_handoff_bytes(value: GraphHandoffV2) -> bytes:
    """Canonical private signature input, never a public/gateway payload."""
    validate_graph_handoff_v2(value)
    return canonical_bytes("graph-handoff-v2", value)


def graph_handoff_signature(value: GraphHandoffV2) -> str:
    validate_graph_handoff_v2(value)
    return digest("graph-handoff-v2", value)


__all__ = [
    "GraphCoverageReason",
    "GraphCoverageV2",
    "GraphCandidateRefV2",
    "GraphHandoffV2",
    "validate_graph_handoff_v2",
    "canonical_graph_handoff_bytes",
    "graph_handoff_signature",
]
