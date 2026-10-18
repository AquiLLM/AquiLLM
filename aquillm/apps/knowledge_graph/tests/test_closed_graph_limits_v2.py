"""Validated graph limits stop unpublished work under live source authority."""

from contextlib import contextmanager
from dataclasses import asdict, replace
from time import monotonic

import pytest

from apps.documents.services.source_deadline import _DEADLINE
from apps.knowledge_graph.tests.test_closed_graph_attempt_v2 import attempt_for
from apps.knowledge_graph.tests.test_graph_candidate_operation_work import graph_frame
from lib.retrieval.turn_budget import TurnBudget, TurnLimits

pytestmark = pytest.mark.django_db(transaction=True)


def returned_envelope(monkeypatch, *, empty=False, reasons=(), after=None):
    from apps.knowledge_graph.retrieval import graph_runtime_v2 as runtime

    original, seen = runtime._run_graph_v2, []

    def run(**kwargs):
        envelope = original(**kwargs)
        coverage = replace(
            envelope.handoff.coverage,
            status="partial" if reasons else "complete",
            reasons=reasons,
        )
        envelope = replace(
            envelope,
            handoff=replace(
                envelope.handoff,
                coverage=coverage,
                candidates=() if empty else envelope.handoff.candidates,
            ),
        )
        seen.append(envelope)
        if after:
            after(kwargs, envelope)
        return envelope

    monkeypatch.setattr(runtime, "_run_graph_v2", run)
    return seen


def finishing_spies(monkeypatch):
    from apps.knowledge_graph.retrieval import graph_candidate_operation_v2 as owner
    from apps.knowledge_graph.retrieval import ready_materialization_v2 as materializer
    from lib import replay_observation

    calls = []
    for module, name in (
        (materializer, "_materialize_rows"),
        (owner, "deferred_baseline"),
        (owner._GraphSourceStage, "publish"),
        (replay_observation, "graph_selected_sources_v2"),
    ):
        original = getattr(module, name)

        def spy(*args, _name=name, _original=original, **kwargs):
            calls.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(module, name, spy)
    return calls


@pytest.mark.parametrize("empty", [False, True])
def test_exact_return_at_owner_deadline_stops_every_finishing_step(monkeypatch, empty):
    now = [0.0]
    budget = TurnBudget(TurnLimits(), clock=lambda: now[0])
    with graph_frame(monkeypatch, budget=budget, timeout_ms=10_000) as active:
        _, graph, source, scope, snapshot, _, operation, _ = active
        epoch = monotonic()
        graph.clock = lambda: epoch + now[0]
        parent = (100.0, lambda: now[0], budget)
        token = _DEADLINE.set(parent)
        before = (source.cache.copy(), scope.bodies.copy(), scope.fusion_v2)
        seen = returned_envelope(
            monkeypatch,
            empty=empty,
            after=lambda kw, e: now.__setitem__(0, kw["deadline"] - epoch),
        )
        calls = finishing_spies(monkeypatch)
        try:
            outcome = attempt_for(active, snapshot, "What is A?").run_native_closure(
                closure_reserve_ms=1250
            )
            assert outcome.kind == "failure" and outcome.pool is None
            assert calls == []
            assert outcome.diagnostics["graph_status"] == "error"
            assert outcome.diagnostics["graph_failure_reason"] == "deadline"
            assert outcome.diagnostics["graph_coverage_v2"] == asdict(
                seen[0].handoff.coverage
            )
            assert outcome.diagnostics["frontier_budget"] == asdict(
                seen[0].frontier_budget
            )
            assert outcome.diagnostics["frontier_invocations"] == [
                asdict(seen[0].frontier_invocation)
            ]
            assert (source.cache, scope.bodies, scope.fusion_v2) == before
            assert operation.active() and operation.deadline == 10.0
            assert _DEADLINE.get() is parent and not outcome.admit_deferred
        finally:
            _DEADLINE.reset(token)


@pytest.mark.parametrize(
    "reasons,want",
    [
        (("deadline",), "deadline"),
        (("frontier_budget",), "frontier_budget"),
        (("deadline", "frontier_budget"), "deadline"),
        (("evidence_budget",), None),
        (("not_ready",), None),
        ((), None),
    ],
)
def test_live_empty_limit_fails_but_ordinary_empty_miss_still_finishes(
    monkeypatch, reasons, want
):
    with graph_frame(monkeypatch) as active:
        seen = returned_envelope(monkeypatch, empty=True, reasons=reasons)
        calls = finishing_spies(monkeypatch)
        outcome = attempt_for(active, active[4], "What is A?").run_native_closure(
            closure_reserve_ms=1250
        )
        assert seen
        if want:
            assert outcome.kind == "failure" and outcome.pool is None and calls == []
            assert outcome.diagnostics["graph_failure_reason"] == want
            assert outcome.admit_deferred
        else:
            assert (
                outcome.kind == "success"
                and outcome.diagnostics["graph_status"] == "miss"
            )
            assert "graph_failure_reason" not in outcome.diagnostics
            assert calls == [
                "_materialize_rows",
                "deferred_baseline",
                "publish",
                "graph_selected_sources_v2",
            ]
        assert outcome.diagnostics["graph_coverage_v2"] == asdict(
            seen[0].handoff.coverage
        )


def test_executor_caught_timeout_returns_real_empty_handoff_and_native_failure(
    monkeypatch,
):
    from apps.knowledge_graph.retrieval import graph_runtime_execution_v2 as execution

    def timeout(**kwargs):
        raise TimeoutError("inner provider allowance exhausted")

    with graph_frame(monkeypatch) as active:
        monkeypatch.setattr(execution, "select_graph_evidence_v2", timeout)
        calls = finishing_spies(monkeypatch)
        outcome = attempt_for(active, active[4], "What is A?").run_native_closure(
            closure_reserve_ms=1250
        )
        assert outcome.kind == "failure" and outcome.pool is None and calls == []
        assert outcome.diagnostics["graph_failure_reason"] == "deadline"
        assert outcome.diagnostics["graph_coverage_v2"]["reasons"] == ("deadline",)
        assert outcome.diagnostics["frontier_budget"]["queries_used"] > 0
        assert active[2].cache == {} and active[3].bodies == {}


def test_live_nonempty_partial_limit_keeps_full_fences_and_success(monkeypatch):
    from apps.knowledge_graph.tests.test_graph_candidate_operation_work import (
        count_full_authorities,
    )

    with graph_frame(monkeypatch) as active:
        proofs = count_full_authorities(monkeypatch)
        returned_envelope(monkeypatch, reasons=("frontier_budget",))
        outcome = attempt_for(active, active[4], "What is A?").run_native_closure(
            closure_reserve_ms=1250
        )
        assert (
            outcome.kind == "success" and outcome.diagnostics["graph_status"] == "hit"
        )
        assert len(proofs) == 2 and len(outcome.pool) == 2
        assert "graph_failure_reason" not in outcome.diagnostics
        assert set(active[3].bodies) == {row.pk for row in outcome.pool}
        assert active[2].budget.sources_used == 2
        assert active[2].budget.text_used["materialized"] == 2


def test_cleanup_deadline_has_no_publication_or_leaked_pending_bodies(monkeypatch):
    from apps.knowledge_graph.retrieval import graph_candidate_operation_v2 as owner

    now = [0.0]
    budget = TurnBudget(TurnLimits(), clock=lambda: now[0])
    with graph_frame(monkeypatch, budget=budget, timeout_ms=10_000) as active:
        epoch = monotonic()
        active[1].clock = lambda: epoch + now[0]
        database, cleanups = owner.bounded_source_database, []

        @contextmanager
        def cleanup(*args):
            with database(*args):
                yield
            cleanups.append(True)
            if len(cleanups) == 2:
                now[0] = 5.0

        monkeypatch.setattr(owner, "bounded_source_database", cleanup)
        calls = finishing_spies(monkeypatch)
        parent = _DEADLINE.get()
        outcome = attempt_for(active, active[4], "What is A?").run_native_closure(
            closure_reserve_ms=1250
        )
        assert outcome.kind == "failure" and outcome.pool is None
        assert outcome.diagnostics["graph_failure_reason"] == "deadline"
        assert calls == ["_materialize_rows", "deferred_baseline"]
        assert active[2].cache == {} and active[3].bodies == {}
        assert active[6].active() and _DEADLINE.get() is parent


@pytest.mark.parametrize("mutation", ["operation", "parent"])
def test_original_context_invalidity_outranks_valid_limit_observations(
    monkeypatch, mutation
):
    now = [0.0]
    budget = TurnBudget(TurnLimits(), clock=lambda: now[0])
    with graph_frame(monkeypatch, budget=budget, timeout_ms=10_000) as active:
        token = _DEADLINE.set((9.0, lambda: now[0], budget))

        def invalidate(kw, envelope):
            if mutation == "operation":
                active[6].live = False
            else:
                now[0] = 9.0

        returned_envelope(
            monkeypatch, empty=True, reasons=("deadline",), after=invalidate
        )
        calls = finishing_spies(monkeypatch)
        try:
            outcome = attempt_for(active, active[4], "What is A?").run_native_closure(
                closure_reserve_ms=1250
            )
            assert outcome.kind == "invalid" and outcome.pool is None and calls == []
            assert outcome.diagnostics == {
                "graph_status": "error",
                "graph_candidate_count": 0,
            }
        finally:
            _DEADLINE.reset(token)
