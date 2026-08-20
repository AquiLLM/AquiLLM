"""Deadline, worker-lifecycle, and typed-failure scheduler tests."""

import threading
import time
from types import SimpleNamespace

import pytest

from apps.knowledge_graph.retrieval.branch_contracts import (
    BranchStatusV1,
    ExtendedBranchFailureReason,
    SharedBranchFailureReason,
)
from apps.knowledge_graph.retrieval.scheduler import HybridGraphBranchScheduler
from apps.knowledge_graph.retrieval.topology.contracts import (
    HybridBranchKind,
    TopologyFailureReason,
)
from apps.knowledge_graph.retrieval.topology.failures import TopologyLoadError
from apps.knowledge_graph.tests.test_retrieval_branch_scheduler import (
    _Runtime,
    _settings,
    _success,
)


def _worker_count() -> int:
    return sum(thread.name.startswith("kg-branch") for thread in threading.enumerate())


def test_shared_preparation_is_bounded_by_the_overall_deadline() -> None:
    release = threading.Event()
    runtime = _Runtime()
    runtime.prepare_shared = lambda **_kwargs: (
        release.wait(timeout=1.0),
        "ready",
    )[1]
    started = time.monotonic()
    try:
        outcome = HybridGraphBranchScheduler(runtime).run(
            query="q",
            baseline=object(),
            authorization=object(),
            settings=_settings(),
            deadline=time.monotonic() + 0.02,
        )
    finally:
        release.set()

    assert time.monotonic() - started < 0.15
    assert outcome.shared_failure_reason is SharedBranchFailureReason.OVERALL_DEADLINE


def test_repeated_timeouts_keep_one_bounded_worker_population() -> None:
    release = threading.Event()
    before = _worker_count()
    try:
        for _ in range(6):
            runtime = _Runtime()
            runtime.run_direct = lambda **_kwargs: (
                release.wait(timeout=1.0),
                _success(HybridBranchKind.DIRECT),
            )[1]
            runtime.prepare_extended = lambda **_kwargs: (
                release.wait(timeout=1.0),
                "prepared",
            )[1]
            HybridGraphBranchScheduler(runtime).run(
                query="q",
                baseline=SimpleNamespace(graph_seeds=(object(),)),
                authorization=object(),
                settings=_settings(direct_ms=10, extended_ms=10),
                deadline=time.monotonic() + 0.02,
            )
        assert _worker_count() - before <= 4
    finally:
        release.set()


def test_typed_local_topology_failure_preserves_completed_sibling() -> None:
    runtime = _Runtime()
    runtime.run_extended = lambda **_kwargs: (_ for _ in ()).throw(
        TopologyLoadError(TopologyFailureReason.EXTENDED_TOPOLOGY_INVALID)
    )

    outcome = HybridGraphBranchScheduler(runtime).run(
        query="q",
        baseline=object(),
        authorization=object(),
        settings=_settings(),
        deadline=time.monotonic() + 0.5,
    )

    assert outcome.direct.status is BranchStatusV1.SUCCEEDED
    assert (
        outcome.extended.failure_reason
        is ExtendedBranchFailureReason.EXTENDED_TOPOLOGY_INVALID
    )
    assert outcome.shared_failure_reason is None


@pytest.mark.parametrize(
    ("topology_reason", "branch_reason"),
    (
        (
            TopologyFailureReason.BACKEND_AUTHENTICATION,
            SharedBranchFailureReason.BACKEND_AUTHENTICATION,
        ),
        (
            TopologyFailureReason.BACKEND_UNAVAILABLE,
            SharedBranchFailureReason.BACKEND_UNAVAILABLE,
        ),
        (
            TopologyFailureReason.BACKEND_PROVENANCE_MISMATCH,
            SharedBranchFailureReason.BACKEND_PROVENANCE_MISMATCH,
        ),
        (
            TopologyFailureReason.BACKEND_SCHEMA_MISMATCH,
            SharedBranchFailureReason.BACKEND_SCHEMA_MISMATCH,
        ),
    ),
)
def test_typed_shared_topology_failure_invalidates_both(topology_reason, branch_reason):
    runtime = _Runtime()
    runtime.run_direct = lambda **_kwargs: (_ for _ in ()).throw(
        TopologyLoadError(topology_reason)
    )

    outcome = HybridGraphBranchScheduler(runtime).run(
        query="q",
        baseline=object(),
        authorization=object(),
        settings=_settings(),
        deadline=time.monotonic() + 0.5,
    )

    assert outcome.shared_failure_reason is branch_reason
    assert outcome.direct.failure_reason is outcome.extended.failure_reason
