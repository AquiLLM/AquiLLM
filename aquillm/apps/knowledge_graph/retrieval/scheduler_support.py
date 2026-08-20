"""Contracts and fixed-result helpers for the hybrid branch scheduler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .branch_contracts import (
    BranchEnvelopeV1,
    BranchSafeDiagnosticsV1,
    BranchStatusV1,
    DirectBranchFailureReason,
    ExtendedBranchFailureReason,
    HybridBranchOutcomeV1,
    SharedBranchFailureReason,
)
from .topology.contracts import HybridBranchKind, TopologyFailureReason


class SharedSchedulerFailure(RuntimeError):
    """A typed fixed failure that invalidates both graph branches."""

    def __init__(self, reason: SharedBranchFailureReason):
        if type(reason) is not SharedBranchFailureReason:
            raise TypeError("reason must be an exact shared branch failure")
        self.reason = reason
        super().__init__(reason.value)


@runtime_checkable
class HybridBranchRuntime(Protocol):
    """Provider adapter whose every operation receives an absolute deadline."""

    def prepare_shared(
        self, *, authorization: object, settings: object, deadline: float
    ) -> object: ...

    def run_direct(
        self,
        *,
        query: str,
        shared: object,
        authorization: object,
        settings: object,
        deadline: float,
    ) -> BranchEnvelopeV1: ...

    def prepare_extended(
        self,
        *,
        baseline: object,
        shared: object,
        authorization: object,
        settings: object,
        deadline: float,
    ) -> object: ...

    def run_extended(
        self,
        *,
        prepared: object,
        shared: object,
        authorization: object,
        settings: object,
        deadline: float,
    ) -> BranchEnvelopeV1: ...


@dataclass(frozen=True, slots=True)
class CompletedBranch:
    envelope: BranchEnvelopeV1
    completed_at: float


def failed_branch(
    kind: HybridBranchKind,
    reason: SharedBranchFailureReason
    | DirectBranchFailureReason
    | ExtendedBranchFailureReason,
    *,
    seed_count: int = 0,
    elapsed_ms: int = 0,
) -> BranchEnvelopeV1:
    return BranchEnvelopeV1(
        kind,
        BranchStatusV1.FAILED,
        None,
        reason,
        BranchSafeDiagnosticsV1(seed_count, 0, 0, 0, elapsed_ms),
    )


def shared_outcome(reason: SharedBranchFailureReason) -> HybridBranchOutcomeV1:
    return HybridBranchOutcomeV1(
        failed_branch(HybridBranchKind.DIRECT, reason),
        failed_branch(HybridBranchKind.EXTENDED, reason),
        reason,
    )


def timeout_branch(
    kind: HybridBranchKind, *, baseline: object, elapsed_ms: int
) -> BranchEnvelopeV1:
    if kind is HybridBranchKind.DIRECT:
        return failed_branch(
            kind,
            DirectBranchFailureReason.EXTRACTOR_TIMEOUT,
            elapsed_ms=elapsed_ms,
        )
    raw_seeds = getattr(baseline, "graph_seeds", ())
    seed_count = min(len(raw_seeds), 64) if type(raw_seeds) is tuple else 0
    if not seed_count:
        return failed_branch(
            kind,
            ExtendedBranchFailureReason.EXTENDED_NO_SEEDS,
            elapsed_ms=elapsed_ms,
        )
    return failed_branch(
        kind,
        ExtendedBranchFailureReason.EXTENDED_TOPOLOGY_TIMEOUT,
        seed_count=seed_count,
        elapsed_ms=elapsed_ms,
    )


def validate_envelope(envelope: object, expected: HybridBranchKind) -> BranchEnvelopeV1:
    if type(envelope) is not BranchEnvelopeV1 or envelope.branch_kind is not expected:
        raise SharedSchedulerFailure(
            SharedBranchFailureReason.BACKEND_PROVENANCE_MISMATCH
        )
    return envelope


_SHARED_TOPOLOGY = {
    TopologyFailureReason.READINESS_MISMATCH: (
        SharedBranchFailureReason.READINESS_MISMATCH
    ),
    TopologyFailureReason.AUTHORIZATION_CONTEXT_INVALID: (
        SharedBranchFailureReason.AUTHORIZATION_CONTEXT_INVALID
    ),
    TopologyFailureReason.BACKEND_AUTHENTICATION: (
        SharedBranchFailureReason.BACKEND_AUTHENTICATION
    ),
    TopologyFailureReason.BACKEND_UNAVAILABLE: (
        SharedBranchFailureReason.BACKEND_UNAVAILABLE
    ),
    TopologyFailureReason.BACKEND_PROVENANCE_MISMATCH: (
        SharedBranchFailureReason.BACKEND_PROVENANCE_MISMATCH
    ),
    TopologyFailureReason.BACKEND_SCHEMA_MISMATCH: (
        SharedBranchFailureReason.BACKEND_SCHEMA_MISMATCH
    ),
    TopologyFailureReason.OVERALL_DEADLINE: SharedBranchFailureReason.OVERALL_DEADLINE,
}
_LOCAL_TOPOLOGY = {
    (
        HybridBranchKind.DIRECT,
        TopologyFailureReason.DIRECT_TOPOLOGY_TIMEOUT,
    ): DirectBranchFailureReason.DIRECT_TOPOLOGY_TIMEOUT,
    (
        HybridBranchKind.DIRECT,
        TopologyFailureReason.DIRECT_TOPOLOGY_INVALID,
    ): DirectBranchFailureReason.DIRECT_TOPOLOGY_INVALID,
    (
        HybridBranchKind.EXTENDED,
        TopologyFailureReason.EXTENDED_TOPOLOGY_TIMEOUT,
    ): ExtendedBranchFailureReason.EXTENDED_TOPOLOGY_TIMEOUT,
    (
        HybridBranchKind.EXTENDED,
        TopologyFailureReason.EXTENDED_TOPOLOGY_INVALID,
    ): ExtendedBranchFailureReason.EXTENDED_TOPOLOGY_INVALID,
}


def map_topology_failure(kind: HybridBranchKind, reason: TopologyFailureReason):
    shared = _SHARED_TOPOLOGY.get(reason)
    if shared is not None:
        return shared
    local = _LOCAL_TOPOLOGY.get((kind, reason))
    if local is None:
        return SharedBranchFailureReason.BACKEND_PROVENANCE_MISMATCH
    return local


__all__ = [
    "CompletedBranch",
    "HybridBranchRuntime",
    "SharedSchedulerFailure",
    "failed_branch",
    "map_topology_failure",
    "shared_outcome",
    "timeout_branch",
    "validate_envelope",
]
