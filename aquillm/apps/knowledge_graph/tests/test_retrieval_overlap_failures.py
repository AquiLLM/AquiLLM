"""Failure and cancellation boundaries of early graph work."""

from concurrent.futures import Future
from threading import Event
from time import monotonic

import pytest

from apps.knowledge_graph.retrieval import scheduler_overlap
from apps.knowledge_graph.retrieval.branch_contracts import (
    BranchStatusV1,
    DirectBranchFailureReason,
    SharedBranchFailureReason,
)
from apps.knowledge_graph.retrieval.scheduler import HybridGraphBranchScheduler
from apps.knowledge_graph.retrieval.scheduler_support import SharedSchedulerFailure
from apps.knowledge_graph.tests.test_retrieval_branch_scheduler import (
    _Runtime,
    _settings,
)


def test_direct_expiring_between_ready_publication_and_collection_is_local(monkeypatch):
    release, ready = Event(), Event()

    class ReadyFuture(Future):
        def set_result(self, result):
            super().set_result(result)
            ready.set()
            assert release.wait(1)

    monkeypatch.setattr(scheduler_overlap, "Future", ReadyFuture)
    clock = [10.0]
    scheduler = HybridGraphBranchScheduler(_Runtime(), clock=lambda: clock[0])
    collect = scheduler._collect
    with scheduler.start(
        query="q", authorization=object(), settings=_settings(), deadline=11.0
    ) as handle:
        assert ready.wait(1)
        clock[0] = 10.2

        def collect_after_early_finishes(**kwargs):
            release.set()
            handle._early.result(timeout=1)
            return collect(**kwargs)

        monkeypatch.setattr(scheduler, "_collect", collect_after_early_finishes)
        outcome = handle.finish(baseline=object(), deadline=11.2)
    assert outcome.direct.failure_reason is DirectBranchFailureReason.EXTRACTOR_TIMEOUT
    assert outcome.extended.status is BranchStatusV1.SUCCEEDED


@pytest.mark.parametrize("phase", ["prepare_shared", "run_direct", "prepare_extended"])
def test_shared_failure_invalidates_both_overlapping_branches(phase):
    runtime = _Runtime()

    def fail(**kwargs):
        raise SharedSchedulerFailure(SharedBranchFailureReason.READINESS_MISMATCH)

    setattr(runtime, phase, fail)
    with HybridGraphBranchScheduler(runtime).start(
        query="secret",
        authorization=object(),
        settings=_settings(),
        deadline=monotonic() + 1.0,
    ) as handle:
        outcome = handle.finish(baseline=object(), deadline=monotonic() + 1.0)
    assert outcome.shared_failure_reason is SharedBranchFailureReason.READINESS_MISMATCH
    assert outcome.direct.failure_reason is outcome.extended.failure_reason


def test_readiness_wait_does_not_renew_expired_early_deadline():
    release = Event()
    runtime = _Runtime()
    runtime.prepare_shared = lambda **kwargs: release.wait(1)
    clock = [10.0]
    with HybridGraphBranchScheduler(runtime, clock=lambda: clock[0]).start(
        query="q", authorization=object(), settings=_settings(), deadline=11.0
    ) as handle:
        clock[0] = 12.0
        outcome = handle.finish(baseline=object(), deadline=13.0)
    release.set()
    assert outcome.shared_failure_reason is SharedBranchFailureReason.OVERALL_DEADLINE


def test_post_baseline_wait_stays_within_existing_overall_deadline():
    release = Event()
    runtime = _Runtime()
    runtime.prepare_extended = lambda **kwargs: release.wait(1)
    with HybridGraphBranchScheduler(runtime).start(
        query="q",
        authorization=object(),
        settings=_settings(),
        deadline=monotonic() + 1.0,
    ) as handle:
        handle._early.result(timeout=1)
        started = monotonic()
        try:
            outcome = handle.finish(baseline=object(), deadline=started + 0.02)
        finally:
            release.set()
            handle._extended.result(timeout=1)
    assert monotonic() - started < 0.15
    assert outcome.direct.status is BranchStatusV1.SUCCEEDED
    assert outcome.extended.status is BranchStatusV1.FAILED


def test_invalid_finish_deadline_still_cancels_unstarted_direct():
    entered, release = Event(), Event()
    runtime = _Runtime()

    def shared(**kwargs):
        entered.set()
        release.wait(1)
        return "ready"

    runtime.prepare_shared = shared
    handle = HybridGraphBranchScheduler(runtime).start(
        query="q",
        authorization=object(),
        settings=_settings(),
        deadline=monotonic() + 1.0,
    )
    assert entered.wait(1)
    try:
        with pytest.raises(ValueError):
            handle.finish(baseline=object(), deadline=0.0)
    finally:
        release.set()
        handle._early.result(timeout=1)
    assert runtime.calls == []
