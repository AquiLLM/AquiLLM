"""Private graph failures leave recovery to their caller under original authority."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from apps.documents.services.hybrid_graph_v2 import hybrid_graph_candidate_pool_v2
from apps.documents.services.source_deadline import _DEADLINE
from apps.documents.tests.test_graph_expired_fallback import install_failure
from apps.documents.tests.test_graph_source_admission import capture, pressure_rows
from apps.knowledge_graph.tests.test_graph_candidate_operation_work import graph_frame
from lib.retrieval.turn_budget import TurnBudget, TurnLimits

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.fixture(autouse=True)
def full_candidate_limit(monkeypatch):
    from django.conf import settings

    monkeypatch.setattr(settings, "RAG_CANDIDATE_MULTIPLIER", 3.0, raising=False)
    monkeypatch.setattr(settings, "RAG_SHORT_QUERY_CANDIDATE_SCALE", 1.0, raising=False)


def diagnostics(status, count=0):
    return {"graph_status": status, "graph_candidate_count": count}


def attempt_for(active, snapshot, query="Atlas"):
    from apps.knowledge_graph.retrieval import graph_candidate_operation_v2 as owner

    f, _, _, _, _, dependencies, _, _ = active
    factory = getattr(owner, "_capture_closed_graph_attempt", None)
    if factory is not None:
        return factory(
            snapshot, query, f.authorization, dependencies, diagnostics=diagnostics
        )

    # Behavioral RED adapter: existing public work exposes the redundant fallback.
    def run(**kwargs):
        rows, observation = hybrid_graph_candidate_pool_v2(
            snapshot, query, f.authorization, dependencies
        )
        return SimpleNamespace(kind="failure", pool=rows, diagnostics=observation)

    return SimpleNamespace(run=run, run_native_closure=run)


def test_private_failure_does_not_read_public_baseline_or_publish_graph(monkeypatch):
    now = [0.0]
    budget = TurnBudget(TurnLimits(), clock=lambda: now[0])
    with graph_frame(
        monkeypatch,
        prepare_sources=pressure_rows,
        dataset=True,
        budget=budget,
        timeout_ms=10_000,
    ) as active:
        f, graph, source, scope, _, dependencies, operation, _ = active
        graph.clock = lambda: now[0]
        snapshot = capture(active)
        before = (source.cache.copy(), scope.bodies.copy(), budget.text_used.copy())
        _, fetched, events = install_failure(monkeypatch, active, now, expired=False)
        outcome = attempt_for(active, snapshot).run(downstream_reserve_ms=0)
        assert fetched == [], (
            "private graph failure must not perform public fallback reads"
        )
        assert outcome.kind == "failure" and outcome.pool is None
        assert operation.active() and events == [] and budget.sources_used == 25
        assert (source.cache, scope.bodies, budget.text_used) == before

        rows, observation = hybrid_graph_candidate_pool_v2(
            snapshot, "Atlas", f.authorization, dependencies
        )
        assert observation["graph_status"] == "error"
        assert observation["source_admission"]["fallback_top_up_count"] == 20
        assert len(rows) == 45 and len(fetched) == 20 and events == []


@pytest.mark.parametrize(
    "cap,elapsed,parent_ms,want_ms",
    [
        (5000, 0, 1500, 250),
        (100, 0, 1500, 100),
        (5000, 100, 1500, 150),
        (5000, 0, 1250, 0),
    ],
)
def test_native_reserve_partitions_effective_parent_after_authorization_cleanup(
    monkeypatch, cap, elapsed, parent_ms, want_ms
):
    from apps.knowledge_graph.retrieval import graph_candidate_operation_v2 as owner

    now = [0.0]
    budget = TurnBudget(TurnLimits(), clock=lambda: now[0])
    with graph_frame(monkeypatch, budget=budget, timeout_ms=4000) as active:
        _, graph, _, _, snapshot, dependencies, operation, _ = active
        graph.clock = lambda: 1000 + now[0]
        dependencies.settings.graph_overall_timeout_ms = cap
        parent = (1_000_000 + parent_ms / 1000, lambda: 1_000_000 + now[0], budget)
        token = _DEADLINE.set(parent)
        original_deadline = operation.deadline
        seen, cleanups = [], []
        original_database, original_run = (
            owner.bounded_source_database,
            owner._run_closed_graph,
        )

        @contextmanager
        def database(*args):
            with original_database(*args):
                yield
            if not cleanups:
                now[0] += elapsed / 1000
            cleanups.append(True)

        def run(*args, **kwargs):
            seen.append(((args[6] - graph.clock()) * 1000, _DEADLINE.get()))
            return original_run(*args, **kwargs)

        monkeypatch.setattr(owner, "bounded_source_database", database)
        monkeypatch.setattr(owner, "_run_closed_graph", run)
        install_failure(monkeypatch, active, now, expired=False)
        try:
            outcome = attempt_for(active, snapshot, "What is A?").run_native_closure(
                closure_reserve_ms=1250
            )
            assert operation.active() and operation.deadline == original_deadline
            assert _DEADLINE.get() is parent
            if want_ms == 0:
                assert seen == [], (
                    "depleted closure allowance must do no graph proof work"
                )
            else:
                assert len(seen) == 1
                allowance, child = seen[0]
                assert want_ms - 2 <= allowance <= want_ms + 0.001
                assert child[1] is parent[1]
                assert (child[0] - child[1]()) * 1000 <= want_ms + 0.001
            assert outcome.kind == "failure" and outcome.pool is None
        finally:
            _DEADLINE.reset(token)


@pytest.mark.parametrize(
    "first,second",
    [("run", "run"), ("run", "run_native_closure"), ("run_native_closure", "run")],
)
def test_attempt_is_consumed_across_both_entry_methods(monkeypatch, first, second):
    from apps.documents.services.representation_contracts import SourceScopeUnavailable

    with graph_frame(monkeypatch) as active:
        snapshot = active[4]
        graph_clock, clock_calls = active[1].clock, []

        def clock():
            clock_calls.append(True)
            return graph_clock()

        active[1].clock = clock
        attempt = attempt_for(active, snapshot, "What is A?")
        _, fetched, events = install_failure(monkeypatch, active, [0.0], expired=False)

        def kwargs(name):
            key = "downstream_reserve_ms" if name == "run" else "closure_reserve_ms"
            return {key: 0}

        getattr(attempt, first)(**kwargs(first))
        before = (fetched.copy(), events.copy(), active[2].budget.text_used.copy())
        clock_count = len(clock_calls)
        with pytest.raises(SourceScopeUnavailable):
            getattr(attempt, second)(**kwargs(second))
        assert (fetched, events, active[2].budget.text_used) == before
        assert len(clock_calls) == clock_count


@pytest.mark.parametrize("method", ["run", "run_native_closure"])
@pytest.mark.parametrize("reserve", [-1, True, 0.5])
def test_invalid_reserve_is_rejected_before_any_work(monkeypatch, method, reserve):
    with graph_frame(monkeypatch) as active:
        attempt = attempt_for(active, active[4], "What is A?")
        _, fetched, events = install_failure(monkeypatch, active, [0.0], expired=False)
        argument = "downstream_reserve_ms" if method == "run" else "closure_reserve_ms"
        with pytest.raises(ValueError):
            getattr(attempt, method)(**{argument: reserve})
        assert fetched == [] and events == [] and active[2].budget.sources_used == 0


@pytest.mark.parametrize(
    "mutation", ["expired_operation", "replaced_operation", "expired_parent"]
)
def test_captured_original_loss_is_invalid_without_graph_work(monkeypatch, mutation):
    from lib.retrieval.operation import _OPERATION, Operation

    now = [0.0]
    budget = TurnBudget(TurnLimits(), clock=lambda: now[0])
    with graph_frame(monkeypatch, budget=budget, timeout_ms=4000) as active:
        _, _, source, scope, snapshot, _, operation, driver = active
        parent = (1001.5, lambda: 1000 + now[0], budget)
        deadline_token = _DEADLINE.set(parent)
        operation_token = None
        try:
            attempt = attempt_for(active, snapshot, "What is A?")
            if mutation == "expired_operation":
                now[0] = 4.0
            elif mutation == "expired_parent":
                now[0] = 1.5
            else:
                operation_token = _OPERATION.set(
                    Operation(budget, deadline=10, clock=lambda: now[0])
                )
            outcome = attempt.run_native_closure(closure_reserve_ms=1250)
            assert outcome.kind == "invalid" and outcome.pool is None
            assert driver.calls == [] and budget.sources_used == 0
            assert source.cache == {} and scope.bodies == {}
            assert operation.deadline == 4.0 and parent[0] == 1001.5
        finally:
            if operation_token is not None:
                _OPERATION.reset(operation_token)
            _DEADLINE.reset(deadline_token)


@pytest.mark.parametrize("fail_after_hydration", [False, True])
def test_private_native_hydration_preserves_proofs_and_publication_boundary(
    monkeypatch, fail_after_hydration
):
    from apps.knowledge_graph.tests.test_graph_candidate_operation_races import (
        after_hydration,
    )
    from apps.knowledge_graph.tests.test_graph_candidate_operation_work import (
        count_full_authorities,
    )
    from lib import replay_observation

    with graph_frame(monkeypatch) as active:
        f, _, source, scope, snapshot, _, _, driver = active
        proofs, events = count_full_authorities(monkeypatch), []
        monkeypatch.setattr(
            replay_observation, "graph_selected_sources_v2", events.append
        )
        if fail_after_hydration:

            def fail():
                assert source.budget.sources_used == 2
                assert not source.cache and not scope.bodies
                raise ValueError("graph failed after real native hydration")

            after_hydration(monkeypatch, fail)
        outcome = attempt_for(active, snapshot, "What is A?").run_native_closure(
            closure_reserve_ms=1250
        )
        assert driver.calls and source.budget.sources_used == 2
        assert source.budget.text_used["materialized"] == 2
        if fail_after_hydration:
            assert outcome.kind == "failure" and outcome.pool is None
            assert not source.cache and not scope.bodies and events == []
        else:
            assert (
                outcome.kind == "success"
                and outcome.diagnostics["graph_status"] == "hit"
            )
            assert {r.pk for r in outcome.pool} == {f.chunks["A"].pk, f.chunks["B"].pk}
            assert len(proofs) == 2 and events
            assert set(scope.bodies) == {r.pk for r in outcome.pool}
