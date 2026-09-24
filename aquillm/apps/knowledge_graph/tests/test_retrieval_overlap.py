"""Request-local overlap and phase deadline contracts."""

import threading
import time
from types import SimpleNamespace

import pytest

from apps.knowledge_graph.retrieval.branch_contracts import (
    BranchStatusV1,
    SharedBranchFailureReason,
)
from apps.knowledge_graph.retrieval.scheduler import HybridGraphBranchScheduler
from apps.knowledge_graph.retrieval.topology.contracts import HybridBranchKind
from apps.knowledge_graph.tests.test_retrieval_branch_scheduler import (
    _Runtime,
    _settings,
    _success,
)


def test_direct_runs_before_baseline_and_extended_receives_finished_snapshot():
    entered = threading.Event()
    release = threading.Event()
    runtime = _Runtime()

    def direct(**kwargs):
        entered.set()
        assert release.wait(1)
        return _success(HybridBranchKind.DIRECT)

    runtime.run_direct = direct
    scheduler = HybridGraphBranchScheduler(runtime)
    with scheduler.start(
        query="q",
        authorization=object(),
        settings=_settings(),
        deadline=time.monotonic() + 1.0,
    ) as handle:
        assert entered.wait(1)
        assert not any(call[0] == "extended-prep" for call in runtime.calls)
        snapshot = SimpleNamespace(graph_seeds=(object(),))
        release.set()
        outcome = handle.finish(baseline=snapshot, deadline=time.monotonic() + 1.0)
    assert outcome.direct.status is BranchStatusV1.SUCCEEDED
    assert outcome.extended.status is BranchStatusV1.SUCCEEDED
    assert (
        next(call[1][1] for call in runtime.calls if call[0] == "extended-prep")
        is snapshot
    )


def test_extended_budget_starts_after_baseline_and_early_direct_is_retained():
    clock = [10.0]
    done = threading.Event()
    runtime = _Runtime()

    def direct(**kwargs):
        done.set()
        return _success(HybridBranchKind.DIRECT)

    runtime.run_direct = direct
    scheduler = HybridGraphBranchScheduler(runtime, clock=lambda: clock[0])
    with scheduler.start(
        query="q", authorization=object(), settings=_settings(), deadline=11.0
    ) as handle:
        assert done.wait(1)
        handle._early.result(timeout=1)
        clock[0] = 10.5
        outcome = handle.finish(baseline=object(), deadline=11.5)
    assert outcome.direct.status is BranchStatusV1.SUCCEEDED
    assert outcome.extended.status is BranchStatusV1.SUCCEEDED
    assert (
        next(call[2] for call in runtime.calls if call[0] == "extended-prep") == 10.725
    )


def test_baseline_failure_cancels_pending_direct_and_releases_capacity():
    entered, release = threading.Event(), threading.Event()
    runtime = _Runtime()

    def shared(**kwargs):
        entered.set()
        assert release.wait(1)
        return "ready"

    runtime.prepare_shared = shared
    with pytest.raises(ValueError, match="baseline failed"):
        with HybridGraphBranchScheduler(runtime).start(
            query="q",
            authorization=object(),
            settings=_settings(),
            deadline=time.monotonic() + 1.0,
        ) as handle:
            assert entered.wait(1)
            raise ValueError("baseline failed")
    release.set()
    handle._early.result(timeout=1)
    assert runtime.calls == []
    with pytest.raises(ValueError, match="already finished"):
        handle.finish(baseline=object(), deadline=time.monotonic() + 1.0)


def test_saturated_pool_never_queues_readiness_or_waits_on_another_worker():
    from apps.knowledge_graph.retrieval.scheduler_workers import BRANCH_WORKERS

    release = threading.Event()
    entered = threading.Barrier(5, timeout=1)
    handles = []

    def shared(**kwargs):
        entered.wait()
        release.wait(1)
        return "ready"

    try:
        for _ in range(4):
            runtime = _Runtime()
            runtime.prepare_shared = shared
            handles.append(
                HybridGraphBranchScheduler(runtime).start(
                    query="q",
                    authorization=object(),
                    settings=_settings(),
                    deadline=time.monotonic() + 1.0,
                )
            )
        entered.wait()
        with HybridGraphBranchScheduler(_Runtime()).start(
            query="q",
            authorization=object(),
            settings=_settings(),
            deadline=time.monotonic() + 1.0,
        ) as rejected:
            outcome = rejected.finish(
                baseline=object(), deadline=time.monotonic() + 1.0
            )
        assert (
            outcome.shared_failure_reason
            is SharedBranchFailureReason.BACKEND_UNAVAILABLE
        )
    finally:
        for handle in handles:
            handle.close()
        release.set()
        for handle in handles:
            handle._early.result(timeout=1)
    # Reuse all four slots after cancellation; no hidden worker or queue remains.
    futures = BRANCH_WORKERS.submit_batch(tuple((lambda: None, ()) for _ in range(4)))
    assert futures is not None
    for future in futures:
        future.result(timeout=1)


@pytest.mark.parametrize(
    "completed_at, expected",
    [(10.125, BranchStatusV1.SUCCEEDED), (10.126, BranchStatusV1.FAILED)],
)
def test_direct_completion_boundary_is_checked_even_after_late_baseline(
    completed_at, expected
):
    clock = [10.0]
    runtime = _Runtime()

    def direct(**kwargs):
        clock[0] = completed_at
        return _success(HybridBranchKind.DIRECT)

    runtime.run_direct = direct
    with HybridGraphBranchScheduler(runtime, clock=lambda: clock[0]).start(
        query="q", authorization=object(), settings=_settings(), deadline=11.0
    ) as handle:
        handle._early.result(timeout=1)
        clock[0] = 12.0
        outcome = handle.finish(baseline=object(), deadline=13.0)
    assert outcome.direct.status is expected
    assert outcome.extended.status is BranchStatusV1.SUCCEEDED


def test_late_shared_readiness_cannot_borrow_post_baseline_budget():
    clock = [10.0]
    runtime = _Runtime()

    def shared(**kwargs):
        clock[0] = 11.1
        return "ready"

    runtime.prepare_shared = shared
    with HybridGraphBranchScheduler(runtime, clock=lambda: clock[0]).start(
        query="q", authorization=object(), settings=_settings(), deadline=11.0
    ) as handle:
        with pytest.raises(Exception):
            handle._early.result(timeout=1)
        outcome = handle.finish(baseline=object(), deadline=12.0)
    assert outcome.shared_failure_reason is SharedBranchFailureReason.OVERALL_DEADLINE
    assert runtime.calls == []


def test_concurrent_request_handles_keep_baselines_and_authorization_separate():
    runtimes = [_Runtime(), _Runtime()]
    auths, baselines = [object(), object()], [object(), object()]
    handles = [
        HybridGraphBranchScheduler(runtime).start(
            query=str(i),
            authorization=auths[i],
            settings=_settings(),
            deadline=time.monotonic() + 1.0,
        )
        for i, runtime in enumerate(runtimes)
    ]
    for i, handle in enumerate(handles):
        with handle:
            outcome = handle.finish(
                baseline=baselines[i], deadline=time.monotonic() + 1.0
            )
            sequential = HybridGraphBranchScheduler(_Runtime()).run(
                query=str(i),
                authorization=auths[i],
                baseline=baselines[i],
                settings=_settings(),
                deadline=time.monotonic() + 1.0,
            )
            assert outcome == sequential
        assert (
            next(call[1] for call in runtimes[i].calls if call[0] == "shared")
            is auths[i]
        )
        assert (
            next(call[1][1] for call in runtimes[i].calls if call[0] == "extended-prep")
            is baselines[i]
        )
