"""Actual public eager source reads keep pending cache private through final fences."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from time import monotonic

import pytest

from apps.documents.services import source_loading
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from lib.retrieval.operation import _OPERATION, Operation, operation_scope
from lib.retrieval.turn_budget import TurnBudget, TurnLimits

from .representation_support import retained as retained
from .representation_support import upload as upload
from .test_source_body_closed_operation import bodies as bodies
from .test_source_body_closed_operation import capture, commit_source_change, query

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.mark.parametrize(
    "change", ["permission", "other_owner", "other_generation", "collision"]
)
def test_actual_final_failure_does_not_leave_positive_public_cache(
    bodies, monkeypatch, change
):
    case = bodies
    original = source_loading._source_body_rows
    observed = []

    def read(*args, **kwargs):
        rows = original(*args, **kwargs)
        assert len(rows) == 8 and case.runtime.budget.sources_used == 8
        observed.append(tuple(row.pk for row in rows))
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(commit_source_change, case, change).result(timeout=10)
        return rows

    with source_loading.source_runtime_scope(case.runtime):
        capture(case)
        monkeypatch.setattr(source_loading, "_source_body_rows", read)
        error = ValueError if change == "collision" else SourceScopeUnavailable
        with pytest.raises(error):
            source_loading.source_query_rows(query(case))
        assert observed == [tuple(row.pk for row in case.chunks)]
        assert case.runtime.cache == {}
        assert case.runtime.budget.text_used["materialized"] == sum(
            len(row.content) for row in case.chunks
        )


@pytest.mark.parametrize(
    "change",
    [
        "budget",
        "cache",
        "authority",
        "snapshot",
        "local_snapshot",
        "runtime_context",
        "operation_context",
        "deadline_context",
    ],
)
def test_post_read_identity_changes_cannot_publish_or_move_charges(
    bodies, monkeypatch, change
):
    from apps.documents.services.source_deadline import _DEADLINE

    case = bodies
    original = source_loading._source_body_rows
    budget, cache, tokens = case.runtime.budget, case.runtime.cache, []
    replacement = TurnBudget(TurnLimits())

    def read(*args, **kwargs):
        rows = original(*args, **kwargs)
        assert len(rows) == 8 and budget.sources_used == 8
        if change == "budget":
            case.runtime.budget = replacement
        elif change == "cache":
            case.runtime.cache = {}
        elif change == "authority":
            case.runtime.authorization = replace(case.runtime.authorization)
        elif change == "snapshot":
            case.runtime._frozen_retrieval_scope = replace(
                case.runtime._frozen_retrieval_scope
            )
        elif change == "local_snapshot":
            args[1].representation = replace(args[1].representation)
        elif change == "runtime_context":
            tokens.append(
                (
                    source_loading._RUNTIME,
                    source_loading._RUNTIME.set(
                        source_loading.SourceRuntime(
                            replacement, case.runtime.authorization
                        )
                    ),
                )
            )
        elif change == "operation_context":
            tokens.append(
                (
                    _OPERATION,
                    _OPERATION.set(
                        Operation(
                            budget, deadline=operation.deadline, clock=operation.clock
                        )
                    ),
                )
            )
        else:
            tokens.append(
                (_DEADLINE, _DEADLINE.set((monotonic() + 100, monotonic, budget)))
            )
        return rows

    with (
        source_loading.source_runtime_scope(case.runtime),
        operation_scope(budget, timeout_ms=10000) as operation,
    ):
        capture(case)
        monkeypatch.setattr(source_loading, "_source_body_rows", read)
        try:
            with pytest.raises(ValueError):
                source_loading.source_query_rows(query(case))
            assert cache == {} and case.runtime.cache == {}
            assert budget.sources_used == 8 and replacement.sources_used == 0
            assert budget.text_used["materialized"] == sum(
                len(row.content) for row in case.chunks
            )
        finally:
            for variable, token in reversed(tokens):
                variable.reset(token)


def test_database_cleanup_precedes_public_cache_commit(bodies, monkeypatch):
    case = bodies
    original, cache, changed = (
        source_loading.bounded_source_database,
        case.runtime.cache,
        [],
    )

    @contextmanager
    def database(*args):
        with original(*args):
            yield
        if case.runtime.budget.sources_used == 8 and not changed:
            changed.append(True)
            case.runtime.cache = {}

    with source_loading.source_runtime_scope(case.runtime):
        capture(case)
        monkeypatch.setattr(source_loading, "bounded_source_database", database)
        with pytest.raises(SourceScopeUnavailable):
            source_loading.source_query_rows(query(case))
        assert changed == [True] and not cache and not case.runtime.cache
        assert case.runtime.budget.sources_used == 8


def test_ledger_clock_callback_cannot_replace_public_cache_at_commit(
    bodies, monkeypatch
):
    case = bodies
    budget, cache, gate, changed = case.runtime.budget, case.runtime.cache, [], []
    real_clock, real_publish = budget._clock, budget.publish

    def clock():
        if gate and not changed:
            changed.append(True)
            case.runtime.cache = {}
        return real_clock()

    def publish(callback):
        if budget.sources_used == 8:
            gate.append(True)
        try:
            return real_publish(callback)
        finally:
            gate.clear()

    with source_loading.source_runtime_scope(case.runtime):
        capture(case)
        monkeypatch.setattr(budget, "_clock", clock)
        monkeypatch.setattr(budget, "publish", publish)
        with pytest.raises(SourceScopeUnavailable):
            source_loading.source_query_rows(query(case))
        assert changed == [True] and not cache and not case.runtime.cache
        assert budget.sources_used == 8


def test_rejected_refresh_preserves_previously_accepted_cache(bodies, monkeypatch):
    with source_loading.source_runtime_scope(bodies.runtime):
        capture(bodies)
        accepted = source_loading.source_query_rows(query(bodies, 2))
        cache = bodies.runtime.cache.copy()
        original = source_loading._source_body_rows

        def read(*args, **kwargs):
            rows = original(*args, **kwargs)
            assert len(rows) == 8
            with ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(commit_source_change, bodies, "permission").result(
                    timeout=10
                )
            return rows

        monkeypatch.setattr(source_loading, "_source_body_rows", read)
        with pytest.raises(SourceScopeUnavailable):
            source_loading.source_query_rows(query(bodies), refresh=True)
        assert bodies.runtime.cache == cache
        assert len(cache) == len(accepted) == 2
        assert bodies.runtime.budget.sources_used == 8
