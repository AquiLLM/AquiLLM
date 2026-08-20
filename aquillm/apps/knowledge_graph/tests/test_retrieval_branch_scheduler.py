from __future__ import annotations

import json
import random
import threading
import time
from hashlib import sha256
from types import SimpleNamespace

from apps.knowledge_graph.retrieval.branch_contracts import (
    BranchEnvelopeV1,
    BranchProvenanceV1,
    BranchSafeDiagnosticsV1,
    BranchStatusV1,
    DirectBranchFailureReason,
    DirectBranchResultV1,
    ExtendedBranchFailureReason,
    ExtendedBranchResultV1,
    GraphBranchCandidateV1,
    SharedBranchFailureReason,
    branch_candidate_order_checksum,
)
from apps.knowledge_graph.retrieval.scheduler import (
    HybridGraphBranchScheduler,
    SharedSchedulerFailure,
)
from apps.knowledge_graph.retrieval.topology.contracts import HybridBranchKind

K = tuple(character * 64 for character in "123456789abcdef")


def _success(kind: HybridBranchKind) -> BranchEnvelopeV1:
    candidate = GraphBranchCandidateV1(
        "a" * 64 if kind is HybridBranchKind.DIRECT else "b" * 64,
        1,
        0.9 if kind is HybridBranchKind.DIRECT else 0.8,
    )
    candidates = (candidate,)
    provenance = BranchProvenanceV1(
        kind,
        K[0],
        K[1],
        1,
        K[2],
        K[3],
        1,
        branch_candidate_order_checksum(candidates),
        125,
        5,
    )
    result_type = (
        DirectBranchResultV1
        if kind is HybridBranchKind.DIRECT
        else ExtendedBranchResultV1
    )
    return BranchEnvelopeV1(
        kind,
        BranchStatusV1.SUCCEEDED,
        result_type(candidates, provenance),
        None,
        BranchSafeDiagnosticsV1(1, 2, 1, 1, 5),
    )


def _local_failure(kind: HybridBranchKind) -> BranchEnvelopeV1:
    reason = (
        DirectBranchFailureReason.DIRECT_NO_SEEDS
        if kind is HybridBranchKind.DIRECT
        else ExtendedBranchFailureReason.EXTENDED_NO_SEEDS
    )
    return BranchEnvelopeV1(
        kind,
        BranchStatusV1.FAILED,
        None,
        reason,
        BranchSafeDiagnosticsV1(0, 0, 0, 0, 3),
    )


def _settings(direct_ms: int = 125, extended_ms: int = 225):
    return SimpleNamespace(
        graph_direct_timeout_ms=direct_ms,
        graph_extended_timeout_ms=extended_ms,
    )


class _Runtime:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object, float]] = []
        self.barrier: threading.Barrier | None = None
        self.direct = _success(HybridBranchKind.DIRECT)
        self.extended = _success(HybridBranchKind.EXTENDED)

    def prepare_shared(self, *, authorization, settings, deadline):
        self.calls.append(("shared", authorization, deadline))
        return "ready"

    def run_direct(self, *, query, shared, authorization, settings, deadline):
        self.calls.append(("direct", query, deadline))
        if self.barrier is not None:
            self.barrier.wait(timeout=1.0)
        return self.direct

    def prepare_extended(self, *, baseline, shared, authorization, settings, deadline):
        self.calls.append(("extended-prep", baseline, deadline))
        if self.barrier is not None:
            self.barrier.wait(timeout=1.0)
        return ("seed", baseline)

    def run_extended(self, *, prepared, shared, authorization, settings, deadline):
        self.calls.append(("extended", prepared, deadline))
        return self.extended


def test_branches_run_concurrently_and_extended_alone_depends_on_baseline() -> None:
    runtime = _Runtime()
    runtime.barrier = threading.Barrier(2)
    baseline, authorization = object(), object()
    outcome = HybridGraphBranchScheduler(runtime, clock=lambda: 100.0).run(
        query="private query",
        baseline=baseline,
        authorization=authorization,
        settings=_settings(),
        deadline=101.0,
    )
    assert outcome.direct.status is outcome.extended.status is BranchStatusV1.SUCCEEDED
    assert runtime.calls[0] == ("shared", authorization, 101.0)
    observed = {name: (value, deadline) for name, value, deadline in runtime.calls[1:]}
    assert observed == {
        "direct": ("private query", 100.125),
        "extended-prep": (baseline, 100.225),
        "extended": (("seed", baseline), 100.225),
    }


def test_branch_local_failure_preserves_completed_sibling() -> None:
    runtime = _Runtime()
    runtime.direct = _local_failure(HybridBranchKind.DIRECT)
    outcome = HybridGraphBranchScheduler(runtime, clock=lambda: 100.0).run(
        query="q",
        baseline=object(),
        authorization=object(),
        settings=_settings(),
        deadline=101.0,
    )
    assert outcome.direct.failure_reason is DirectBranchFailureReason.DIRECT_NO_SEEDS
    assert outcome.extended.status is BranchStatusV1.SUCCEEDED
    assert outcome.shared_failure_reason is None


def test_shared_failure_cancels_peer_without_waiting_or_exposing_completion() -> None:
    release = threading.Event()
    runtime = _Runtime()

    def blocked(**kwargs):
        release.wait(timeout=1.0)
        return _success(HybridBranchKind.DIRECT)

    def unavailable(**kwargs):
        raise SharedSchedulerFailure(SharedBranchFailureReason.BACKEND_UNAVAILABLE)

    runtime.run_direct = blocked
    runtime.run_extended = unavailable
    started = time.monotonic()
    try:
        outcome = HybridGraphBranchScheduler(runtime).run(
            query="q",
            baseline=object(),
            authorization=object(),
            settings=_settings(),
            deadline=time.monotonic() + 0.5,
        )
    finally:
        release.set()
    assert time.monotonic() - started < 0.2
    assert (
        outcome.shared_failure_reason is SharedBranchFailureReason.BACKEND_UNAVAILABLE
    )
    assert outcome.direct.failure_reason is outcome.extended.failure_reason


def test_independent_budget_times_out_one_branch_and_keeps_completed_peer() -> None:
    release = threading.Event()
    runtime = _Runtime()

    def blocked(**kwargs):
        release.wait(timeout=1.0)
        return _success(HybridBranchKind.DIRECT)

    runtime.run_direct = blocked
    started = time.monotonic()
    try:
        outcome = HybridGraphBranchScheduler(runtime).run(
            query="q",
            baseline=SimpleNamespace(graph_seeds=(object(),)),
            authorization=object(),
            settings=_settings(direct_ms=20, extended_ms=100),
            deadline=time.monotonic() + 0.5,
        )
    finally:
        release.set()
    assert time.monotonic() - started < 0.15
    assert outcome.direct.failure_reason is DirectBranchFailureReason.EXTRACTOR_TIMEOUT
    assert outcome.extended.status is BranchStatusV1.SUCCEEDED
    assert outcome.shared_failure_reason is None


def test_shared_preparation_failure_fails_both_without_starting_branches() -> None:
    runtime = _Runtime()

    def invalid(**kwargs):
        raise SharedSchedulerFailure(
            SharedBranchFailureReason.AUTHORIZATION_CONTEXT_INVALID
        )

    runtime.prepare_shared = invalid
    outcome = HybridGraphBranchScheduler(runtime, clock=lambda: 100.0).run(
        query="q",
        baseline=object(),
        authorization=object(),
        settings=_settings(),
        deadline=101.0,
    )
    assert outcome.shared_failure_reason is (
        SharedBranchFailureReason.AUTHORIZATION_CONTEXT_INVALID
    )
    assert not runtime.calls


def _digest(outcome) -> str:
    payload = {
        "direct": [
            outcome.direct.status.value,
            getattr(outcome.direct.failure_reason, "value", None),
            [row.chunk_key for row in outcome.direct.result.candidates]
            if outcome.direct.result
            else [],
        ],
        "extended": [
            outcome.extended.status.value,
            getattr(outcome.extended.failure_reason, "value", None),
            [row.chunk_key for row in outcome.extended.result.candidates]
            if outcome.extended.result
            else [],
        ],
        "shared": getattr(outcome.shared_failure_reason, "value", None),
    }
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_completion_orders_fixed_seeds_zero_through_nineteen_have_one_digest() -> None:
    digests = set()
    for seed in range(20):
        randomizer = random.Random(seed)
        delays = [randomizer.random() / 1000 for _ in range(2)]
        runtime = _Runtime()
        direct, extended = runtime.run_direct, runtime.run_extended
        runtime.run_direct = lambda **kwargs: (time.sleep(delays[0]), direct(**kwargs))[
            1
        ]
        runtime.run_extended = lambda **kwargs: (
            time.sleep(delays[1]),
            extended(**kwargs),
        )[1]
        outcome = HybridGraphBranchScheduler(runtime).run(
            query="q",
            baseline=object(),
            authorization=object(),
            settings=_settings(),
            deadline=time.monotonic() + 0.5,
        )
        digests.add(_digest(outcome))
    assert digests == {
        "07b4b7cda435da868c13f9c98ed41a3ae1b74c219f049fb350097fc01f188ea4"
    }
