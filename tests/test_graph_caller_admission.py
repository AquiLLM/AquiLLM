"""Captured caller admission across real scheduler delegates without ContextVars."""

from concurrent.futures import Future
from contextvars import Context
from unittest.mock import patch

from lib.retrieval.operation import operation_scope
from lib.retrieval.turn_budget import TurnBudget, TurnLimits
from apps.knowledge_graph.retrieval.scheduler import HybridGraphBranchScheduler
from apps.knowledge_graph.retrieval.branch_contracts import SharedBranchFailureReason
from apps.knowledge_graph.tests.test_retrieval_branch_scheduler import _Runtime, _settings


class ImmediateAbsentContextWorkers:
    def __init__(self, before=lambda function: None):
        self.before = before

    def submit(self, function, *args):
        future = Future()
        self.before(function)
        try:
            future.set_result(Context().run(function, *args))
        except BaseException as error:
            future.set_exception(error)
        return future

    def submit_batch(self, requests):
        return tuple(self.submit(function, *args) for function, args in requests)


def test_revocation_during_shared_preparation_does_not_admit_new_branches():
    budget = TurnBudget(TurnLimits(), clock=lambda: 100.0)
    runtime = _Runtime()
    scheduler = HybridGraphBranchScheduler(runtime, clock=lambda: 100.0)
    with operation_scope(budget, timeout_ms=1000) as bound:
        prepare = runtime.prepare_shared
        def revoke(**kwargs):
            result = prepare(**kwargs)
            bound.live = False
            return result
        runtime.prepare_shared = revoke
        with patch("apps.knowledge_graph.retrieval.scheduler.BRANCH_WORKERS",
                   ImmediateAbsentContextWorkers()):
            outcome = scheduler.run(query="q", baseline=object(), authorization=object(),
                settings=_settings(), deadline=101.0, caller_bound=bound)
    assert outcome.shared_failure_reason is SharedBranchFailureReason.OVERALL_DEADLINE
    assert [call[0] for call in runtime.calls] == ["shared"]


def test_overlap_rechecks_revocation_after_waiting_for_readiness():
    budget = TurnBudget(TurnLimits(), clock=lambda: 100.0)
    runtime = _Runtime()
    scheduler = HybridGraphBranchScheduler(runtime, clock=lambda: 100.0)
    with operation_scope(budget, timeout_ms=1000) as bound:
        with patch("apps.knowledge_graph.retrieval.scheduler_overlap.BRANCH_WORKERS",
                   ImmediateAbsentContextWorkers()):
            handle = scheduler.start(query="q", authorization=object(),
                settings=_settings(), deadline=101.0, caller_bound=bound)
            original_result = handle._ready.result
            def revoked_result(**kwargs):
                result = original_result(**kwargs)
                bound.live = False
                return result
            handle._ready.result = revoked_result
            outcome = handle.finish(baseline=object(), deadline=101.0)
    assert outcome.shared_failure_reason is SharedBranchFailureReason.OVERALL_DEADLINE
    assert [call[0] for call in runtime.calls] == ["shared", "direct"]


def test_shared_worker_entry_rechecks_revocation_after_submission():
    budget = TurnBudget(TurnLimits(), clock=lambda: 100.0)
    runtime = _Runtime()
    with operation_scope(budget, timeout_ms=1000) as bound:
        workers = ImmediateAbsentContextWorkers(lambda function: setattr(bound, "live", False))
        with patch("apps.knowledge_graph.retrieval.scheduler.BRANCH_WORKERS", workers):
            outcome = HybridGraphBranchScheduler(runtime, clock=lambda: 100.0).run(
                query="q", baseline=object(), authorization=object(), settings=_settings(),
                deadline=101.0, caller_bound=bound)
    assert outcome.shared_failure_reason is SharedBranchFailureReason.OVERALL_DEADLINE
    assert runtime.calls == []


def test_direct_worker_entry_rechecks_revocation_after_submission():
    budget = TurnBudget(TurnLimits(), clock=lambda: 100.0)
    runtime = _Runtime()
    with operation_scope(budget, timeout_ms=1000) as bound:
        def revoke_direct(function):
            if function.__name__ == "_direct":
                bound.live = False
        with patch("apps.knowledge_graph.retrieval.scheduler.BRANCH_WORKERS",
                ImmediateAbsentContextWorkers(revoke_direct)):
            outcome = HybridGraphBranchScheduler(runtime, clock=lambda: 100.0).run(
                query="q", baseline=object(), authorization=object(), settings=_settings(),
                deadline=101.0, caller_bound=bound)
    assert outcome.shared_failure_reason is SharedBranchFailureReason.OVERALL_DEADLINE
    assert [call[0] for call in runtime.calls] == ["shared"]


def test_extended_execution_rechecks_revocation_after_preparation():
    budget = TurnBudget(TurnLimits(), clock=lambda: 100.0)
    runtime = _Runtime()
    with operation_scope(budget, timeout_ms=1000) as bound:
        prepare = runtime.prepare_extended
        def revoke(**kwargs):
            result = prepare(**kwargs)
            bound.live = False
            return result
        runtime.prepare_extended = revoke
        with patch("apps.knowledge_graph.retrieval.scheduler.BRANCH_WORKERS",
                ImmediateAbsentContextWorkers()):
            outcome = HybridGraphBranchScheduler(runtime, clock=lambda: 100.0).run(
                query="q", baseline=object(), authorization=object(), settings=_settings(),
                deadline=101.0, caller_bound=bound)
    assert outcome.shared_failure_reason is SharedBranchFailureReason.OVERALL_DEADLINE
    assert [call[0] for call in runtime.calls] == ["shared", "direct", "extended-prep"]


def test_overlap_extended_worker_entry_rechecks_revocation_after_submission():
    budget = TurnBudget(TurnLimits(), clock=lambda: 100.0)
    runtime = _Runtime()
    with operation_scope(budget, timeout_ms=1000) as bound:
        def revoke_extended(function):
            if function.__name__ == "_extended":
                bound.live = False
        with patch("apps.knowledge_graph.retrieval.scheduler_overlap.BRANCH_WORKERS",
                ImmediateAbsentContextWorkers(revoke_extended)):
            handle = HybridGraphBranchScheduler(runtime, clock=lambda: 100.0).start(
                query="q", authorization=object(), settings=_settings(),
                deadline=101.0, caller_bound=bound)
            outcome = handle.finish(baseline=object(), deadline=101.0)
    assert outcome.shared_failure_reason is SharedBranchFailureReason.OVERALL_DEADLINE
    assert [call[0] for call in runtime.calls] == ["shared", "direct"]
