"""Exact context identity, deadline and custom callback boundary controls."""

from contextlib import contextmanager
from copy import copy

import pytest

from apps.knowledge_graph.tests.test_graph_candidate_operation_races import (
    after_hydration,
)
from apps.knowledge_graph.tests.test_graph_candidate_operation_work import (
    count_full_authorities,
    graph_frame,
    run_graph,
)
from lib.retrieval.turn_budget import TurnBudget, TurnLimits

pytestmark = pytest.mark.django_db(transaction=True)


def test_class_hook_before_first_owner_import_cannot_enter_private_path(monkeypatch):
    import importlib
    import sys
    from types import SimpleNamespace

    from apps.knowledge_graph.retrieval.production_runtime import (
        ProductionHybridBranchRuntime,
    )

    original = ProductionHybridBranchRuntime.run_v2
    monkeypatch.setattr(
        ProductionHybridBranchRuntime, "run_v2", lambda self, **kw: original(self, **kw)
    )
    name = "apps.knowledge_graph.retrieval.graph_candidate_operation_v2"
    parent = importlib.import_module("apps.knowledge_graph.retrieval")
    monkeypatch.delattr(parent, "graph_candidate_operation_v2", raising=False)
    monkeypatch.delitem(sys.modules, name, raising=False)
    owner = importlib.import_module(name)
    graph = object.__new__(ProductionHybridBranchRuntime)
    graph.settings = SimpleNamespace(graph_remediation_mode="v2")
    monkeypatch.setattr(
        owner, "current_scope", lambda: SimpleNamespace(using="default")
    )

    def forbidden(*args):
        pytest.fail("class wrapper must use the independently guarded public path")

    monkeypatch.setattr(owner, "_closed_header", forbidden)
    dependencies = SimpleNamespace(runtime=graph, settings=graph.settings)
    assert (
        owner._try_closed_graph_pool(
            None, None, None, dependencies, diagnostics=None, fallback=None
        )
        is None
    )


def test_caller_dependency_replacement_is_rejected_before_captured_lock(monkeypatch):
    with graph_frame(monkeypatch) as active:
        _, graph, source, _, _, dependencies, _, _ = active
        inner, entered, changed = source.lock, [], []

        class Lock:
            def __enter__(self):
                entered.append(True)
                return inner.__enter__()

            def __exit__(self, *args):
                return inner.__exit__(*args)

        source.lock = Lock()
        original = source.budget.remaining_ms

        def remaining():
            if not changed:
                changed.append(True)
                object.__setattr__(dependencies, "runtime", copy(graph))
            return original()

        monkeypatch.setattr(source.budget, "remaining_ms", remaining)
        rows, diagnostics = run_graph(active)
        assert changed == [True] and not entered
        assert rows == () and diagnostics["graph_status"] == "error"
        assert source.budget.sources_used == 0 and not source.cache


@pytest.mark.parametrize("when", ["lock", "database", "cleanup"])
def test_dependency_replacement_during_context_entry_or_cleanup_never_publishes(
    monkeypatch, when
):
    from apps.knowledge_graph.retrieval import graph_candidate_operation_v2 as owner

    with graph_frame(monkeypatch) as active:
        _, graph, source, scope, _, dependencies, _, _ = active
        entered, mutated = [], []

        def mutate():
            mutated.append(True)
            object.__setattr__(dependencies, "settings", copy(graph.settings))

        original = owner.bounded_source_database

        @contextmanager
        def database(*args):
            entered.append(True)
            with original(*args):
                if when == "database":
                    mutate()
                yield
            if when == "cleanup":
                mutate()

        monkeypatch.setattr(owner, "bounded_source_database", database)
        if when == "lock":
            inner = source.lock

            class Lock:
                def __enter__(self):
                    result = inner.__enter__()
                    mutate()
                    return result

                def __exit__(self, *args):
                    return inner.__exit__(*args)

            source.lock = Lock()
        rows, diagnostics = run_graph(active)
        assert mutated == [True]
        assert rows == () and diagnostics["graph_status"] == "error"
        assert not source.cache and not scope.bodies
        if when == "lock":
            assert not entered
        assert source.budget.sources_used == (2 if when == "cleanup" else 0)


def test_original_action_expiry_rejects_while_operation_live_and_global_time_remains(
    monkeypatch,
):
    clock = [0.0]
    budget = TurnBudget(TurnLimits(), clock=lambda: clock[0])
    with graph_frame(monkeypatch, budget=budget, timeout_ms=10000) as active:
        _, _, source, scope, _, _, operation, _ = active

        def expire():
            assert operation.live
            clock[0] = 10.0
            assert operation.live and budget.remaining_ms() == 5000

        after_hydration(monkeypatch, expire)
        rows, diagnostics = run_graph(active)
        assert rows == () and diagnostics["graph_status"] == "error"
        assert budget.sources_used == 2
        assert operation.live and not source.cache and not scope.bodies


def test_same_budget_live_operation_replacement_cannot_adopt_staged_graph_bodies(
    monkeypatch,
):
    from lib.retrieval.operation import _OPERATION, Operation

    with graph_frame(monkeypatch, timeout_ms=10000) as active:
        _, _, source, scope, _, _, original, _ = active
        token = []

        def replace_operation():
            replacement = Operation(
                source.budget, deadline=original.deadline, clock=original.clock
            )
            token.append(_OPERATION.set(replacement))
            assert original.active() and replacement.active()

        after_hydration(monkeypatch, replace_operation)
        try:
            rows, diagnostics = run_graph(active)
            assert rows == () and diagnostics["graph_status"] == "error"
            assert not source.cache and not scope.bodies
            assert source.budget.sources_used == 2
        finally:
            if token:
                _OPERATION.reset(token[0])


def test_original_ledger_clock_callback_cannot_replace_cache_at_publication(
    monkeypatch,
):
    with graph_frame(monkeypatch) as active:
        _, _, source, scope, _, _, _, _ = active
        calls = count_full_authorities(monkeypatch)
        original_cache, original_clock, original_publish = (
            source.cache,
            source.budget._clock,
            source.budget.publish,
        )
        gate, changed = [], []

        def clock():
            if gate and not changed:
                changed.append(True)
                source.cache = {}
            return original_clock()

        def publish(callback):
            if len(calls) == 2:
                gate.append(True)
            try:
                return original_publish(callback)
            finally:
                gate.clear()

        monkeypatch.setattr(source.budget, "_clock", clock)
        monkeypatch.setattr(source.budget, "publish", publish)
        rows, diagnostics = run_graph(active)
        assert changed == [True]
        assert rows == () and diagnostics["graph_status"] == "error"
        assert not original_cache and not source.cache and not scope.bodies
        assert source.budget.sources_used == 2


def test_original_parent_source_deadline_cannot_be_hidden_by_other_clock_epoch(
    monkeypatch,
):
    from time import monotonic

    from apps.documents.services.source_deadline import _DEADLINE

    with graph_frame(monkeypatch) as active:
        _, _, source, scope, _, _, _, _ = active
        clock = [monotonic() + 1_000_000]
        original_deadline = clock[0] + 100
        token = _DEADLINE.set((original_deadline, lambda: clock[0], source.budget))

        def expire_parent():
            clock[0] = original_deadline

        after_hydration(monkeypatch, expire_parent)
        try:
            rows, diagnostics = run_graph(active)
            assert rows == () and diagnostics["graph_status"] == "error"
            assert not source.cache and not scope.bodies
            assert source.budget.sources_used == 2
        finally:
            _DEADLINE.reset(token)


def test_graph_phase_expiry_preserves_fresh_baseline_with_original_global_time(
    monkeypatch,
):
    from time import monotonic

    with graph_frame(monkeypatch) as active:
        f, graph, source, _, _, _, operation, _ = active
        offset = [0]
        graph.clock = lambda: monotonic() + offset[0]

        def expire_graph():
            offset[0] = 6
            assert operation.active() and source.budget.remaining_ms() > 0

        after_hydration(monkeypatch, expire_graph)
        rows, diagnostics = run_graph(active)
        assert tuple(row.pk for row in rows) == (f.chunks["A"].pk,)
        assert diagnostics["graph_status"] == "error"
        assert not source.cache and source.budget.sources_used == 2


@pytest.mark.parametrize("epoch", [0, 1_000_000_000_000])
def test_live_parent_other_clock_epochs_allow_exact_native_graph_success(
    monkeypatch, epoch
):
    from time import monotonic

    from apps.documents.services.source_deadline import _DEADLINE

    with graph_frame(monkeypatch) as active:
        _, _, source, scope, _, _, _, _ = active
        origin = monotonic()

        def clock():
            return monotonic() - origin + epoch

        token = _DEADLINE.set((clock() + 10, clock, source.budget))
        try:
            calls = count_full_authorities(monkeypatch)
            rows, diagnostics = run_graph(active)
            assert len(rows) == 2 and diagnostics["graph_status"] == "hit"
            assert len(calls) == 2 and len(source.cache) == 2
            assert set(scope.bodies) == {row.pk for row in rows}
        finally:
            _DEADLINE.reset(token)
