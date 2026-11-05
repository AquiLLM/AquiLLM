"""Observing a failure cannot redecide its already established recovery result."""

from time import monotonic

import pytest

from apps.knowledge_graph.tests.test_closed_graph_attempt_v2 import attempt_for
from apps.knowledge_graph.tests.test_graph_candidate_operation_work import graph_frame
from lib.retrieval.turn_budget import TurnBudget, TurnLimits

pytestmark = pytest.mark.django_db(transaction=True)


def test_logging_callback_does_not_redecide_deferred_recovery(monkeypatch):
    from apps.knowledge_graph.retrieval import graph_failure_observations_v2 as diagnostic
    from apps.knowledge_graph.retrieval import ready_materialization_v2 as materializer

    now = [0.0]
    budget = TurnBudget(TurnLimits(), clock=lambda: now[0])
    with graph_frame(monkeypatch, budget=budget, timeout_ms=10000) as active:
        epoch = monotonic()
        active[1].clock = lambda: epoch + now[0]

        def fail(**kwargs):
            raise ValueError("private provider failure")

        class Logger:
            def warning(self, event, **fields):
                now[0] = 6.0  # Graph allowance ends at five; original owner at ten.

        monkeypatch.setattr(materializer, "_materialize_closed_rows", fail)
        monkeypatch.setattr(diagnostic, "logger", Logger())
        outcome = attempt_for(active, active[4], "What is A?").run_native_closure(
            closure_reserve_ms=1250)
        assert now[0] == 6.0 and active[6].active()
        assert outcome.kind == "failure" and outcome.admit_deferred is True
        assert outcome.pool is None and active[2].cache == {} and active[3].bodies == {}
        assert budget.sources_used == 2 and "graph_failure_reason" not in outcome.diagnostics
