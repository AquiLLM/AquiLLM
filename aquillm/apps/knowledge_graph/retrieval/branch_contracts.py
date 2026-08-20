# ruff: noqa: E501
# fmt: off
"""Immutable contracts for independently failing hybrid graph branches."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, fields
from enum import StrEnum
from hashlib import sha256
from math import isfinite
from typing import final

from .topology.contracts import (
    HybridBranchKind,
    ProjectedSeedV1,
    ReadyGenerationBundleV1,
    TopologyCapsV1,
    TopologyDeadlineV1,
    projected_seed_checksum,
)

_KEY = re.compile(r"[0-9a-f]{64}")
class BranchStatusV1(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
class SharedBranchFailureReason(StrEnum):
    READINESS_MISMATCH = "readiness_mismatch"
    AUTHORIZATION_CONTEXT_INVALID = "authorization_context_invalid"
    BACKEND_AUTHENTICATION = "backend_authentication"
    BACKEND_UNAVAILABLE = "backend_unavailable"
    BACKEND_PROVENANCE_MISMATCH = "backend_provenance_mismatch"
    BACKEND_SCHEMA_MISMATCH = "backend_schema_mismatch"
    OVERALL_DEADLINE = "overall_deadline"
    FUSION_INVALID = "fusion_invalid"
class DirectBranchFailureReason(StrEnum):
    EXTRACTOR_TIMEOUT = "extractor_timeout"
    EXTRACTOR_AUTH = "extractor_auth"
    EXTRACTOR_PROVENANCE = "extractor_provenance"
    MIXED_ONTOLOGY = "mixed_ontology"
    DIRECT_SEED_INVALID = "direct_seed_invalid"
    DIRECT_NO_SEEDS = "direct_no_seeds"
    DIRECT_EMBEDDING_UNAVAILABLE = "direct_embedding_unavailable"
    DIRECT_TOPOLOGY_TIMEOUT = "direct_topology_timeout"
    DIRECT_TOPOLOGY_INVALID = "direct_topology_invalid"
    DIRECT_PPR_INVALID = "direct_ppr_invalid"
class ExtendedBranchFailureReason(StrEnum):
    EXTENDED_SEED_INVALID = "extended_seed_invalid"
    EXTENDED_NO_SEEDS = "extended_no_seeds"
    EXTENDED_TOPOLOGY_TIMEOUT = "extended_topology_timeout"
    EXTENDED_TOPOLOGY_INVALID = "extended_topology_invalid"
    EXTENDED_PPR_INVALID = "extended_ppr_invalid"
def _key(value: object, name: str) -> None:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact str")
    if _KEY.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase 64-hex opaque value")
def _count(value: object, name: str, maximum: int, minimum: int = 0) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact int")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} is outside its hard cap")
@final
@dataclass(frozen=True, slots=True)
class GraphBranchCandidateV1:
    chunk_key: str
    rank: int
    score: float
    def __post_init__(self) -> None:
        _key(self.chunk_key, "chunk_key")
        _count(self.rank, "rank", 20, 1)
        if type(self.score) is not float:
            raise TypeError("score must be an exact float")
        if not isfinite(self.score) or not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be finite and in [0, 1]")
def branch_candidate_order_checksum(
    candidates: tuple[GraphBranchCandidateV1, ...],
) -> str:
    _candidates(candidates)
    payload = [
        {"chunk_key": row.chunk_key, "rank": row.rank, "score": row.score.hex()}
        for row in candidates
    ]
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return sha256(encoded).hexdigest()
@final
@dataclass(frozen=True, slots=True)
class BranchProvenanceV1:
    branch_kind: HybridBranchKind
    ready_bundle_checksum: str
    seed_checksum: str
    seed_count: int
    topology_snapshot_checksum: str
    ppr_algorithm_signature: str
    candidate_count: int
    candidate_order_checksum: str
    deadline_ms: int
    elapsed_ms: int
    def __post_init__(self) -> None:
        if type(self.branch_kind) is not HybridBranchKind:
            raise TypeError("branch_kind must be exact")
        for name in (
            "ready_bundle_checksum",
            "seed_checksum",
            "topology_snapshot_checksum",
            "ppr_algorithm_signature",
            "candidate_order_checksum",
        ):
            _key(getattr(self, name), name)
        _count(self.seed_count, "seed_count", 64, 1)
        _count(self.candidate_count, "candidate_count", 20)
        _count(self.deadline_ms, "deadline_ms", 5000, 1)
        _count(self.elapsed_ms, "elapsed_ms", self.deadline_ms)
@final
@dataclass(frozen=True, slots=True)
class DirectBranchRequestV1:
    ready: ReadyGenerationBundleV1
    seeds: tuple[ProjectedSeedV1, ...]
    seed_checksum: str
    caps: TopologyCapsV1
    deadline: TopologyDeadlineV1
    def __post_init__(self) -> None:
        _validate_request(self, HybridBranchKind.DIRECT)
@final
@dataclass(frozen=True, slots=True)
class ExtendedBranchRequestV1:
    ready: ReadyGenerationBundleV1
    seeds: tuple[ProjectedSeedV1, ...]
    seed_checksum: str
    caps: TopologyCapsV1
    deadline: TopologyDeadlineV1
    def __post_init__(self) -> None:
        _validate_request(self, HybridBranchKind.EXTENDED)
def _validate_request(value: object, expected: HybridBranchKind) -> None:
    if type(value.ready) is not ReadyGenerationBundleV1:
        raise TypeError("ready must be exact")
    if type(value.caps) is not TopologyCapsV1 or type(value.deadline) is not TopologyDeadlineV1:
        raise TypeError("caps and deadline must be exact")
    if type(value.seeds) is not tuple or any(
        type(seed) is not ProjectedSeedV1 for seed in value.seeds
    ):
        raise TypeError("seeds must contain exact ProjectedSeedV1 values")
    if not value.seeds or len(value.seeds) > value.caps.max_seeds:
        raise ValueError("seeds are empty or exceed caps")
    keys = tuple(seed.identity_key for seed in value.seeds)
    if len(set(keys)) != len(keys) or keys != tuple(sorted(keys)):
        raise ValueError("seeds must be unique and canonically sorted")
    _key(value.seed_checksum, "seed_checksum")
    if value.seed_checksum != projected_seed_checksum(value.seeds):
        raise ValueError("seed_checksum does not bind seeds")
    if (
        value.caps.branch_kind is not expected
        or value.deadline.branch_kind is not expected
    ):
        raise ValueError(f"request must use {expected.value} branch contracts")
@final
@dataclass(frozen=True, slots=True)
class DirectBranchResultV1:
    candidates: tuple[GraphBranchCandidateV1, ...]
    provenance: BranchProvenanceV1
    def __post_init__(self) -> None:
        _validate_result(self, HybridBranchKind.DIRECT)
@final
@dataclass(frozen=True, slots=True)
class ExtendedBranchResultV1:
    candidates: tuple[GraphBranchCandidateV1, ...]
    provenance: BranchProvenanceV1
    def __post_init__(self) -> None:
        _validate_result(self, HybridBranchKind.EXTENDED)
def _validate_result(value: object, expected: HybridBranchKind) -> None:
    _candidates(value.candidates)
    if type(value.provenance) is not BranchProvenanceV1:
        raise TypeError("provenance must be exact")
    if value.provenance.branch_kind is not expected:
        raise ValueError("result provenance has the wrong branch kind")
    if value.provenance.candidate_count != len(value.candidates):
        raise ValueError("candidate count disagrees with provenance")
    if value.provenance.candidate_order_checksum != branch_candidate_order_checksum(
        value.candidates
    ):
        raise ValueError("candidate checksum disagrees with provenance")
@final
@dataclass(frozen=True, slots=True)
class BranchSafeDiagnosticsV1:
    seed_count: int
    node_count: int
    edge_count: int
    candidate_count: int
    elapsed_ms: int
    def __post_init__(self) -> None:
        for field, maximum in zip(fields(self), (64, 200, 1000, 20, 5000), strict=True):
            _count(getattr(self, field.name), field.name, maximum)
type BranchResultV1 = DirectBranchResultV1 | ExtendedBranchResultV1
type BranchFailureReason = (
    SharedBranchFailureReason | DirectBranchFailureReason | ExtendedBranchFailureReason
)
@final
@dataclass(frozen=True, slots=True)
class BranchEnvelopeV1:
    branch_kind: HybridBranchKind
    status: BranchStatusV1
    result: BranchResultV1 | None
    failure_reason: BranchFailureReason | None
    diagnostics: BranchSafeDiagnosticsV1
    def __post_init__(self) -> None:
        if (
            type(self.branch_kind) is not HybridBranchKind
            or type(self.status) is not BranchStatusV1
        ):
            raise TypeError("branch_kind and status must be exact")
        if type(self.diagnostics) is not BranchSafeDiagnosticsV1:
            raise TypeError("diagnostics must be exact")
        result_type = (
            DirectBranchResultV1
            if self.branch_kind is HybridBranchKind.DIRECT
            else ExtendedBranchResultV1
        )
        local_type = (
            DirectBranchFailureReason
            if self.branch_kind is HybridBranchKind.DIRECT
            else ExtendedBranchFailureReason
        )
        if self.status is BranchStatusV1.SUCCEEDED:
            if type(self.result) is not result_type or self.failure_reason is not None:
                raise ValueError(
                    "succeeded status requires its exact result and no failure"
                )
            if (
                self.diagnostics.seed_count != self.result.provenance.seed_count
                or self.diagnostics.candidate_count != len(self.result.candidates)
                or self.diagnostics.elapsed_ms != self.result.provenance.elapsed_ms
            ):
                raise ValueError(
                    "successful diagnostics disagree with result provenance"
                )
        else:
            if self.result is not None:
                raise ValueError("failed status must not expose a result")
            if type(self.failure_reason) not in {local_type, SharedBranchFailureReason}:
                raise TypeError("failure_reason is not valid for this branch")
            if self.diagnostics.candidate_count:
                raise ValueError("failed diagnostics must not claim candidates")
@final
@dataclass(frozen=True, slots=True)
class HybridBranchOutcomeV1:
    direct: BranchEnvelopeV1
    extended: BranchEnvelopeV1
    shared_failure_reason: SharedBranchFailureReason | None
    def __post_init__(self) -> None:
        if (
            type(self.direct) is not BranchEnvelopeV1
            or type(self.extended) is not BranchEnvelopeV1
        ):
            raise TypeError("branch envelopes must be exact")
        if (
            self.direct.branch_kind is not HybridBranchKind.DIRECT
            or self.extended.branch_kind is not HybridBranchKind.EXTENDED
        ):
            raise ValueError("outcome branch envelopes are misassigned")
        shared = self.shared_failure_reason
        if shared is not None and type(shared) is not SharedBranchFailureReason:
            raise TypeError("shared_failure_reason must be exact")
        observed = tuple(
            row.failure_reason
            for row in (self.direct, self.extended)
            if type(row.failure_reason) is SharedBranchFailureReason
        )
        if shared is None and observed:
            raise ValueError("shared envelope failure requires shared outcome reason")
        if shared is not None and observed != (shared, shared):
            raise ValueError(
                "shared failure must fail both branches with the same reason"
            )
def _candidates(value: object) -> None:
    if type(value) is not tuple or any(
        type(row) is not GraphBranchCandidateV1 for row in value
    ):
        raise TypeError("candidates must contain exact GraphBranchCandidateV1 values")
    if len(value) > 20:
        raise ValueError("candidates exceed the hard cap")
    if tuple(row.rank for row in value) != tuple(range(1, len(value) + 1)):
        raise ValueError("candidate ranks must be contiguous and one-based")
    if len({row.chunk_key for row in value}) != len(value):
        raise ValueError("candidate chunk keys must be unique")
