"""Baseline completion keeps the original authority, phase and cumulative debits."""

from contextlib import contextmanager

import pytest

from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.services.source_loading import SourcePreparationLimited
from apps.documents.tests._native_graph_search_closure_support import (
    native_case, run_current_or_native,
)

pytestmark = pytest.mark.django_db


def ready_failure(case, monkeypatch, mode="deadline"):
    from apps.documents.services.source_deadline import _DEADLINE
    from apps.knowledge_graph.retrieval import ready_scope_repository_v2 as ready
    from apps.knowledge_graph.retrieval.ready_authority_reads_v2 import capture_authority_check

    select, failures = ready._select_closed_ready_scope_v2, []

    def fail(**kwargs):
        select(**kwargs)
        check = capture_authority_check()
        bound = _DEADLINE.get()
        failures.append(bound)
        if mode == "early_timeout":
            raise TimeoutError("early graph transport timeout")
        if mode == "transient_context":
            # The graph's context manager restores this before owner validation.
            _DEADLINE.set((bound[0], bound[1], bound[2]))
        case.now[0] += bound[0] - bound[1]() + .001
        if mode == "parent":
            case.now[0] = case.operation.deadline + .001
        elif mode == "permission":
            case.f.policy.rows = ()
        check()
        pytest.fail("real ready-authority check ignored expiry")

    monkeypatch.setattr(ready, "_select_closed_ready_scope_v2", fail)
    return failures


@pytest.mark.parametrize("mode", ["early_timeout", "transient_context", "parent", "permission"])
def test_baseline_completion_requires_witnessed_child_and_live_owner(monkeypatch, mode):
    from apps.documents.services import source_body_batch

    monkeypatch.setenv("RAG_SCOPED_BASELINE_RECOVERY_ENABLED", "0")
    with native_case(monkeypatch, fail=False, controlled_graph_clock=True) as case:
        initial = {row.pk for row in case.snapshot.baseline_candidates}
        before = case.source.budget.text_used.copy()
        failures = ready_failure(case, monkeypatch, mode)
        fetched, fetch = [], source_body_batch._fetch_body_rows

        def read(query):
            rows = fetch(query)
            fetched.extend(row.pk for row in rows)
            return rows

        monkeypatch.setattr(source_body_batch, "_fetch_body_rows", read)
        if mode in {"parent", "permission"}:
            with pytest.raises(SourceScopeUnavailable):
                run_current_or_native(case)
        else:
            result = run_current_or_native(case)
            assert {row.pk for row in result.final_rows[2]} == initial
            assert not result.graph_diagnostics["source_admission"]["baseline_completion_attempted"]
        assert len(failures) == 1 and not fetched
        assert case.source.budget.sources_used == len(initial) == 25
        assert case.source.budget.text_used == before
        assert set(case.scope.bodies) == initial
        assert case.inference == []


@pytest.mark.parametrize("boundary", ["slice", "database_cleanup", "preparation_cleanup"])
def test_baseline_completion_cleanup_does_not_publish_failed_reads(monkeypatch, boundary):
    from apps.documents.services import native_scoped_fallback, source_body_batch
    from apps.documents.services.source_deadline import _DEADLINE

    monkeypatch.setenv("RAG_SCOPED_BASELINE_RECOVERY_ENABLED", "0")
    with native_case(monkeypatch, fail=False, controlled_graph_clock=True) as case:
        initial = {row.pk for row in case.snapshot.baseline_candidates}
        cache = case.source.cache.copy()
        before = case.source.budget.text_used.copy()
        ready_failure(case, monkeypatch)
        fetched, fetch = [], source_body_batch._fetch_body_rows

        def read(query):
            rows = fetch(query)
            fetched.extend(row.pk for row in rows)
            if boundary == "slice":
                deadline, clock, _ = _DEADLINE.get()
                case.now[0] += deadline - clock() + .001
            return rows

        monkeypatch.setattr(source_body_batch, "_fetch_body_rows", read)
        if boundary == "database_cleanup":
            database = case.scope.database

            @contextmanager
            def database_cleanup():
                with database():
                    yield
                if fetched:
                    case.operation.live = False

            monkeypatch.setattr(case.scope, "database", database_cleanup)
        elif boundary == "preparation_cleanup":
            prepare = native_scoped_fallback.source_preparation_scope

            @contextmanager
            def preparation_cleanup(*args):
                with prepare(*args):
                    yield
                case.operation.live = False

            monkeypatch.setattr(native_scoped_fallback, "source_preparation_scope", preparation_cleanup)

        if boundary == "slice":
            result = run_current_or_native(case)
            assert {row.pk for row in result.final_rows[2]} == initial
            assert case.operation.active()
        else:
            error = SourcePreparationLimited if boundary == "preparation_cleanup" else SourceScopeUnavailable
            with pytest.raises(error):
                run_current_or_native(case)
        assert fetched and case.source.budget.sources_used == 45
        assert case.source.budget.text_used["materialized"] > before["materialized"]
        assert set(case.scope.bodies) == initial
        assert case.source.cache == cache and case.inference == []


@pytest.mark.parametrize("exhausted", ["sources", "time", "materialization"])
def test_baseline_completion_keeps_exhausted_allowances_and_actual_charges(monkeypatch, exhausted):
    monkeypatch.setenv("RAG_SCOPED_BASELINE_RECOVERY_ENABLED", "0")
    with native_case(monkeypatch, fail=False, controlled_graph_clock=True) as case:
        budget = case.source.budget
        initial = {row.pk for row in case.snapshot.baseline_candidates}
        if exhausted == "sources":
            case.snapshot.source_admission.top_up()
            assert budget.sources_used == 45
        elif exhausted == "time":
            case.now[0] = case.operation.deadline - 3.5
        else:
            assert budget.reserve_text(
                budget.limits.materialized_codepoints - budget.text_used["materialized"],
                kind="materialized",
            )
        before = budget.text_used.copy()
        bodies, cache = case.scope.bodies.copy(), case.source.cache.copy()
        ready_failure(case, monkeypatch)
        result = run_current_or_native(case)
        assert {row.pk for row in result.final_rows[2]} == initial
        assert budget.text_used == before
        assert case.scope.bodies == bodies and case.source.cache == cache
        assert budget.sources_used == (25 if exhausted == "time" else 45)
        assert case.operation.active() and case.inference == []


def test_baseline_completion_obeys_original_operation_source_ceiling(monkeypatch):
    from apps.documents.services.chunk_search_authorization import authorized_search_rows
    from apps.documents.services.closed_search_finalizer import _bind_search_authorizer
    from apps.documents.tests.test_graph_source_admission import capture
    from lib.retrieval.operation import operation_scope

    monkeypatch.setenv("RAG_SCOPED_BASELINE_RECOVERY_ENABLED", "0")
    with native_case(monkeypatch, fail=False, controlled_graph_clock=True, initial_capture=False) as case:
        with operation_scope(case.source.budget, timeout_ms=6500, source_limit=30) as operation:
            case.operation = operation
            case.snapshot = capture(case.active)
            case.authorize = _bind_search_authorizer(
                authorized_search_rows, case.f.authorization, True,
            )
            assert case.source.budget.sources_used == 10
            ready_failure(case, monkeypatch)
            result = run_current_or_native(case)
            assert len(result.final_rows[2]) == case.source.budget.sources_used == 30
            assert case.source.budget.remaining_source_capacity() == 0
            assert operation.source_limit == 30 and operation.active()
        assert case.source.budget.remaining_source_capacity() == 15


@pytest.mark.parametrize("scoped", [False, True])
def test_only_established_scoped_completion_marks_recovery_origins(monkeypatch, scoped):
    from apps.chat.services import rag_recovery_origins
    from apps.knowledge_graph.tests.test_ready_scope import _DOC_A

    monkeypatch.setenv("RAG_SCOPED_BASELINE_RECOVERY_ENABLED", "1")
    with native_case(
        monkeypatch, fail=False, controlled_graph_clock=True,
        requested=frozenset({_DOC_A}) if scoped else None,
    ) as case:
        initial = {row.pk for row in case.snapshot.baseline_candidates}
        ready_failure(case, monkeypatch)
        recorded, record = [], rag_recovery_origins.record_native_recovery

        def observe(captured, rows, ordinary_ids):
            recorded.append((tuple(row.pk for row in rows), frozenset(ordinary_ids)))
            return record(captured, rows, ordinary_ids)

        monkeypatch.setattr(rag_recovery_origins, "record_native_recovery", observe)
        result = run_current_or_native(case)
        ids = {row.pk for row in result.final_rows[2]}
        assert len(ids) == 45 and initial < ids
        assert len(recorded) == 1 and set(recorded[0][0]) == ids
        assert recorded[0][1] == (initial if scoped else ids)
        assert case.source.budget.sources_used == 45 and case.inference == []
