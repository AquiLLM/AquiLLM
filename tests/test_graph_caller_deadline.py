"""Pure caller-bound tests: no Django setup, database or retrieval transport."""

import asyncio
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from lib.retrieval import operation
from lib.retrieval.turn_budget import TurnBudget, TurnLimits
from apps.knowledge_graph.retrieval.branch_contracts import (
    BranchStatusV1, ExtendedBranchFailureReason, SharedBranchFailureReason,
)
from apps.knowledge_graph.retrieval.scheduler import HybridGraphBranchScheduler
from apps.knowledge_graph.retrieval.scheduler_support import failed_branch
from apps.knowledge_graph.retrieval.topology.contracts import HybridBranchKind
from apps.knowledge_graph.tests.test_retrieval_branch_scheduler import _Runtime, _settings


class CallerBoundTests(unittest.TestCase):
    def setUp(self):
        self.now = [100.0]
        self.budget = TurnBudget(TurnLimits(), clock=lambda: self.now[0])

    def test_capture_is_original_operation_and_translates_different_clock_epochs(self):
        with operation.operation_scope(self.budget, timeout_ms=1000) as bound:
            self.assertIs(operation.capture_operation_bound(self.budget), bound)
            self.now[0] += 0.25
            self.assertAlmostEqual(bound.deadline_for(lambda: 10.0), 10.75)
            self.assertAlmostEqual(bound.deadline_for(lambda: 10.0, reserve_ms=200), 10.55)
            self.assertTrue(self.budget.can_publish())
        self.assertFalse(bound.live)
        self.assertLessEqual(bound.deadline_for(lambda: 10.0), 10.0)
        self.assertGreater(self.budget.remaining_ms(), 10000)

    def test_nested_bound_cannot_extend_parent(self):
        with operation.operation_scope(self.budget, timeout_ms=1000) as parent:
            self.now[0] += 0.5
            with operation.operation_scope(self.budget, timeout_ms=5000) as child:
                self.assertAlmostEqual(child.deadline_for(lambda: 10.0), 10.5)
                parent.live = False
                self.assertFalse(self.budget.can_publish())
                self.assertLessEqual(child.deadline_for(lambda: 10.0), 10.0)

    def test_nested_unbounded_scope_does_not_erase_parent_expiry(self):
        with operation.operation_scope(self.budget, timeout_ms=1000):
            with operation.operation_scope(self.budget) as child:
                self.assertIs(operation.capture_operation_bound(self.budget), child)
                self.now[0] += 1.001
                self.assertFalse(self.budget.can_publish())

    def test_smaller_nested_timeout_wins(self):
        with operation.operation_scope(self.budget, timeout_ms=1000):
            with operation.operation_scope(self.budget, timeout_ms=100) as child:
                self.assertAlmostEqual(child.deadline_for(lambda: 10.0), 10.1)

    def test_wrong_budget_and_expired_capture_fail_closed(self):
        other = TurnBudget(TurnLimits(), clock=lambda: self.now[0])
        with operation.operation_scope(self.budget, timeout_ms=1000):
            with self.assertRaises(ValueError):
                operation.capture_operation_bound(other)
            self.now[0] += 1.001
            with self.assertRaises(TimeoutError):
                operation.capture_operation_bound(self.budget)

    def test_unbounded_operation_retains_revocation(self):
        with operation.operation_scope(self.budget) as original:
            self.assertIsNone(operation.capture_operation_bound(self.budget))
            original.live = False
            self.assertFalse(self.budget.can_publish())
            with self.assertRaises(TimeoutError):
                operation.capture_operation_bound(self.budget)

    def test_global_budget_is_an_additional_minimum(self):
        with operation.operation_scope(self.budget, timeout_ms=30000) as bound:
            self.assertLessEqual(bound.deadline_for(lambda: 10.0), 25.0)

    def test_translation_samples_destination_before_original_clock(self):
        calls = []
        with operation.operation_scope(self.budget, timeout_ms=1000) as bound:
            original_clock = bound.clock
            bound.clock = lambda: (calls.append("source"), original_clock())[1]
            bound.deadline_for(lambda: (calls.append("destination"), 10.0)[1])
        self.assertEqual(calls[0], "destination")

    def test_clock_callback_revocation_cannot_return_live_allowance(self):
        with operation.operation_scope(self.budget, timeout_ms=1000) as bound:
            def revoke():
                bound.live = False
                return self.now[0]
            bound.clock = revoke
            token = operation._OPERATION.set(None)  # Actual graph worker context absence.
            try:
                self.assertLessEqual(bound.deadline_for(lambda: 10.0), 10.0)
            finally:
                operation._OPERATION.reset(token)


class GraphRetentionTests(unittest.TestCase):
    def setUp(self):
        self.source_now, self.graph_now = [100.0], [10.0]
        self.budget = TurnBudget(TurnLimits(), clock=lambda: self.source_now[0])
        self.runtime = _Runtime()
        self.scheduler = HybridGraphBranchScheduler(self.runtime, clock=lambda: self.graph_now[0])

    def start(self, bound):
        return self.scheduler.start(query="q", authorization=object(), settings=_settings(),
            deadline=11.0, caller_bound=bound, downstream_reserve_ms=200)

    def test_elapsed_graph_cutoff_keeps_timely_direct_without_launching_extended(self):
        with operation.operation_scope(self.budget, timeout_ms=1000) as bound:
            with self.start(bound) as handle:
                handle._early.result(timeout=1)
                self.source_now[0], self.graph_now[0] = 100.85, 10.85
                outcome = handle.finish(baseline=object(), deadline=11.85)
            self.assertTrue(self.budget.can_publish())
        self.assertIs(outcome.direct.status, BranchStatusV1.SUCCEEDED)
        self.assertIs(outcome.extended.failure_reason, ExtendedBranchFailureReason.EXTENDED_BRANCH_TIMEOUT)
        self.assertFalse(any(call[0].startswith("extended") for call in self.runtime.calls))
        self.assertLessEqual(self.runtime.calls[0][2], 10.8)

    def test_revocation_with_global_time_remaining_withdraws_completed_direct(self):
        with operation.operation_scope(self.budget, timeout_ms=1000) as bound:
            with self.start(bound) as handle:
                handle._early.result(timeout=1)
                bound.live = False
                outcome = handle.finish(baseline=object(), deadline=11.0)
        self.assertIs(outcome.shared_failure_reason, SharedBranchFailureReason.OVERALL_DEADLINE)
        self.assertIs(outcome.direct.status, BranchStatusV1.FAILED)
        self.assertFalse(any(call[0].startswith("extended") for call in self.runtime.calls))

    def test_action_expiry_never_retains_completed_direct(self):
        with operation.operation_scope(self.budget, timeout_ms=1000) as bound:
            with self.start(bound) as handle:
                handle._early.result(timeout=1)
                self.source_now[0], self.graph_now[0] = 101.001, 11.001
                outcome = handle.finish(baseline=object(), deadline=12.0)
        self.assertIs(outcome.shared_failure_reason, SharedBranchFailureReason.OVERALL_DEADLINE)

    def test_no_bound_existing_post_baseline_extended_contract_is_unchanged(self):
        with self.scheduler.start(query="q", authorization=object(), settings=_settings(),
                deadline=11.0) as handle:
            handle._early.result(timeout=1)
            self.graph_now[0] = 10.5
            outcome = handle.finish(baseline=object(), deadline=11.5)
        self.assertIs(outcome.direct.status, BranchStatusV1.SUCCEEDED)
        self.assertIs(outcome.extended.status, BranchStatusV1.SUCCEEDED)

    def test_shared_corruption_invalidates_completed_sibling_at_elapsed_cutoff(self):
        self.runtime.direct = failed_branch(HybridBranchKind.DIRECT,
            SharedBranchFailureReason.BACKEND_PROVENANCE_MISMATCH)
        with operation.operation_scope(self.budget, timeout_ms=1000) as bound:
            with self.start(bound) as handle:
                handle._early.result(timeout=1)
                self.source_now[0], self.graph_now[0] = 100.85, 10.85
                outcome = handle.finish(baseline=object(), deadline=11.85)
        self.assertIs(outcome.shared_failure_reason, SharedBranchFailureReason.BACKEND_PROVENANCE_MISMATCH)

    def test_direct_completed_after_original_branch_cutoff_is_not_adopted(self):
        original = self.runtime.run_direct
        def late(**kwargs):
            self.source_now[0], self.graph_now[0] = 100.15, 10.15
            return original(**kwargs)
        self.runtime.run_direct = late
        with operation.operation_scope(self.budget, timeout_ms=1000) as bound:
            with self.start(bound) as handle:
                handle._early.result(timeout=1)
                self.source_now[0], self.graph_now[0] = 100.85, 10.85
                outcome = handle.finish(baseline=object(), deadline=11.85)
        self.assertIs(outcome.direct.status, BranchStatusV1.FAILED)


class SourceBoundTests(unittest.TestCase):
    def test_source_default_without_source_context_is_clamped_by_exact_action(self):
        from apps.documents.services.source_deadline import source_remaining_ms
        now = [100.0]
        budget = TurnBudget(TurnLimits(), clock=lambda: now[0])
        with operation.operation_scope(budget, timeout_ms=1000):
            now[0] += 0.25
            self.assertEqual(source_remaining_ms(15000), 750)
            now[0] += 1
            self.assertEqual(source_remaining_ms(15000), 0)

    def test_unbounded_operation_keeps_default_source_allowance(self):
        from apps.documents.services.source_deadline import source_remaining_ms
        budget = TurnBudget(TurnLimits())
        with operation.operation_scope(budget):
            self.assertEqual(source_remaining_ms(1234), 1234)


class AcquisitionBoundTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_action_captures_exact_reserved_timeout_and_revokes_on_return(self):
        from apps.chat.services.rag_acquisition import acquire_evidence
        now = [100.0]
        budget = TurnBudget(TurnLimits(), clock=lambda: now[0])
        now[0] = 104.0  # 11 seconds globally: action must stop at 7, leaving 4.
        captured = []
        sentinel = object()
        async def execute(action):
            captured.append(operation._OPERATION.get())
            return sentinel
        stubs = {
            "lib.llm.turn_context": SimpleNamespace(tool_timeout=lambda **kwargs: 10.0),
            "apps.documents.services.source_loading": SimpleNamespace(current_source_runtime=lambda: None),
            "apps.chat.services.retrieval_query_runtime": SimpleNamespace(
                run_optional_proposal=None, action_observation=lambda *args: {"signature": "q"}),
        }
        with patch.dict(sys.modules, stubs):
            result = await acquire_evidence("q", budget=budget, llm=None,
                execute=execute, evidence_views=lambda rows: ())
        self.assertEqual(result.results, (sentinel,))
        self.assertIs(captured[0].budget, budget)
        self.assertEqual(captured[0].deadline, 111.0)
        self.assertFalse(captured[0].live)
        self.assertEqual(budget.remaining_ms(), 11000)

    async def test_cancellation_revokes_captured_original_with_global_time_remaining(self):
        from apps.chat.services.rag_acquisition import acquire_evidence
        budget = TurnBudget(TurnLimits())
        entered, captured = asyncio.Event(), []
        async def execute(action):
            captured.append(operation._OPERATION.get())
            entered.set()
            await asyncio.Future()
        stubs = {
            "lib.llm.turn_context": SimpleNamespace(tool_timeout=lambda **kwargs: 10.0),
            "apps.documents.services.source_loading": SimpleNamespace(current_source_runtime=lambda: None),
            "apps.chat.services.retrieval_query_runtime": SimpleNamespace(
                run_optional_proposal=None, action_observation=lambda *args: {"signature": "q"}),
        }
        with patch.dict(sys.modules, stubs):
            task = asyncio.create_task(acquire_evidence("q", budget=budget, llm=None,
                execute=execute, evidence_views=lambda rows: ()))
            await asyncio.wait_for(entered.wait(), 1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertIsNotNone(captured[0].deadline)
        self.assertFalse(captured[0].live)
        self.assertEqual(budget.stop_reason, "cancelled")


class GraphConsumerTests(unittest.TestCase):
    def test_early_graph_passes_original_bound_and_existing_transport_reserve(self):
        from apps.documents.services import hybrid_graph_lifecycle as lifecycle
        from apps.documents.services.hybrid_graph_authorization import HybridGraphRetrievalDependencies
        from apps.documents.services import hybrid_graph_deadline
        budget = TurnBudget(TurnLimits())
        dependencies = HybridGraphRetrievalDependencies(_Runtime(), SimpleNamespace(
            graph_remediation_mode="legacy", memgraph_traversal_enabled=True,
            graph_overall_timeout_ms=5000), lambda **kwargs: ())
        with operation.operation_scope(budget, timeout_ms=1000) as bound:
            with patch.object(lifecycle, "is_exact_authorization_context", return_value=True), \
                    patch.object(hybrid_graph_deadline, "captured_graph_bound", return_value=bound), \
                    patch.object(hybrid_graph_deadline, "graph_rerank_reserve_ms", return_value=200), \
                    patch.object(HybridGraphBranchScheduler, "start", return_value="handle") as start:
                self.assertEqual(lifecycle.start_graph("q", object(), dependencies, True), "handle")
        self.assertIs(start.call_args.kwargs.get("caller_bound"), bound)
        self.assertEqual(start.call_args.kwargs.get("downstream_reserve_ms"), 200)

    def test_graph_bound_requires_same_original_runtime_authorization_and_budget(self):
        from apps.documents.services.hybrid_graph_deadline import captured_graph_bound
        budget = TurnBudget(TurnLimits())
        authorization = object()
        runtime = SimpleNamespace(budget=budget, authorization=authorization)
        scope = SimpleNamespace(runtime=runtime, authorization=authorization)
        with operation.operation_scope(budget, timeout_ms=1000) as bound:
            with patch.dict(sys.modules, {"apps.documents.services.chunk_search_scope":
                    SimpleNamespace(current_scope=lambda: scope)}):
                self.assertIs(captured_graph_bound(), bound)
                scope.authorization = object()
                with self.assertRaises(ValueError):
                    captured_graph_bound()
                scope.authorization = authorization
                runtime.budget = TurnBudget(TurnLimits())
                with self.assertRaises(ValueError):
                    captured_graph_bound()

    def test_rerank_reserve_reads_existing_configuration(self):
        import os
        from apps.documents.services.hybrid_graph_deadline import graph_rerank_reserve_ms
        with patch.dict(os.environ, {"APP_RERANK_TIMEOUT_SECONDS": "2"}):
            self.assertEqual(graph_rerank_reserve_ms(), 2000)


if __name__ == "__main__":
    unittest.main()
