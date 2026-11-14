"""Early SQL cancellation cannot authorize new baseline reads or poison cleanup."""

import pytest
from django.db import DatabaseError, connection

from apps.documents.models import TextChunk
from apps.documents.services.graph_source_admission_policy import charged_sources
from apps.documents.services.source_deadline import _DEADLINE
from apps.documents.tests._native_graph_search_closure_support import (
    native_case,
    run_current_or_native,
)
from lib.retrieval.operation import _OPERATION

pytestmark = pytest.mark.django_db(transaction=True)


def _transaction_status():
    driver = connection.connection
    if hasattr(driver, "get_transaction_status"):
        return int(driver.get_transaction_status())
    return int(driver.info.transaction_status)


def test_early_postgres_cancellation_rolls_back_without_new_baseline_publication(monkeypatch):
    """Catch generic-57014 recovery and SQL executed before failed-child rollback."""
    from apps.documents.services import graph_candidate_admission, source_body_batch
    from apps.knowledge_graph.retrieval import ready_scope_repository_v2 as ready
    from lib import replay_observation

    if connection.vendor != "postgresql":
        pytest.skip("requires actual PostgreSQL statement cancellation")
    monkeypatch.setenv("RAG_SCOPED_BASELINE_RECOVERY_ENABLED", "0")
    with native_case(monkeypatch, fail=False) as case:
        source, scope, operation = case.source, case.scope, case.operation
        budget, authorization = source.budget, source.authorization
        baseline = tuple(case.snapshot.baseline_candidates)
        initial_ids = {row.pk for row in baseline}
        assert len(initial_ids) == budget.sources_used == 25
        assert len(scope.requested_document_ids) > 1
        assert scope.requested_document_ids == authorization.selected_document_ids
        assert set(case.snapshot.source_admission.references) - initial_ids
        before = (source.cache.copy(), scope.bodies.copy(), budget.text_used.copy(),
                  charged_sources(budget), operation.deadline, budget.actions_used,
                  budget.pairs_used.copy())
        parent_bound = _DEADLINE.get()
        with connection.cursor() as cursor:
            cursor.execute("SELECT current_setting('statement_timeout')")
            original_timeout = cursor.fetchone()[0]
        select = ready._select_closed_ready_scope_v2
        cancelled, usable_reads, events = [], [], []

        def cancel_after_real_authority(**kwargs):
            select(**kwargs)  # Preserve the real persisted readiness/authority path.
            bound = _DEADLINE.get()
            assert bound is not None and bound[2] is budget
            assert _OPERATION.get() is operation and operation.active()
            assert bound[0] - bound[1]() > 1
            try:
                with connection.cursor() as cursor:
                    cursor.execute("SET LOCAL statement_timeout = '20ms'")
                    cursor.execute("SELECT pg_sleep(1)")
            except DatabaseError as error:
                cause = error.__cause__
                state = getattr(cause, "sqlstate", None) or getattr(cause, "pgcode", None)
                assert state == "57014"
                assert _transaction_status() == 3  # PostgreSQL INERROR, before rollback.
                assert _DEADLINE.get() is bound and bound[0] - bound[1]() > 1
                assert operation.active() and budget.can_publish()
                cancelled.append(state)
                raise  # Keep the actual unmodified database exception ineligible.
            pytest.fail("PostgreSQL did not cancel the graph child statement")

        def observe_after_cancellation(execute, sql, params, many, context):
            if cancelled and not usable_reads and not sql.lstrip().upper().startswith("ROLLBACK"):
                assert _transaction_status() != 3
                assert not connection.needs_rollback
                result = execute(sql, params, many, context)
                usable_reads.append(True)
                return result
            return execute(sql, params, many, context)

        def unexpected_acquisition(*args, **kwargs):
            pytest.fail("early SQL cancellation acquired new baseline bodies or ranks")

        monkeypatch.setattr(ready, "_select_closed_ready_scope_v2", cancel_after_real_authority)
        monkeypatch.setattr(source_body_batch, "_fetch_body_rows", unexpected_acquisition)
        monkeypatch.setattr(graph_candidate_admission, "capture_metadata", unexpected_acquisition)
        monkeypatch.setattr(TextChunk, "get_chunk_embedding", unexpected_acquisition)
        monkeypatch.setattr(replay_observation, "graph_selected_sources_v2", events.append)
        with connection.execute_wrapper(observe_after_cancellation):
            result = run_current_or_native(case)

        assert cancelled == ["57014"] and usable_reads == [True]
        assert result is not None and result.graph_diagnostics["graph_status"] == "error"
        failure = result.graph_diagnostics["graph_failure"]
        assert failure["type"] == "database_error"
        assert failure["reason"] == "statement_cancelled" and failure["owner_valid"] is True
        admission = result.graph_diagnostics["source_admission"]
        assert admission["baseline_completion_attempted"] is False
        assert admission["fallback_top_up_skipped"] is True
        assert admission["fallback_top_up_count"] == 0
        assert {row.pk for row in result.final_rows[2]} == initial_ids
        assert budget.sources_used == 25 and charged_sources(budget) == before[3]
        assert source.cache == before[0] and scope.bodies == before[1]
        assert budget.text_used == before[2]
        assert operation.deadline == before[4] and budget.actions_used == before[5]
        assert budget.pairs_used == before[6]
        assert source.authorization is authorization and scope.runtime is source
        assert _OPERATION.get() is operation and _DEADLINE.get() is parent_bound
        assert operation.active() and budget.can_publish()
        assert not events and not case.inference and result.score_set is None
        assert connection.get_autocommit() and not connection.in_atomic_block
        assert not connection.needs_rollback
        with connection.cursor() as cursor:
            cursor.execute("SELECT current_setting('statement_timeout'), 1")
            assert cursor.fetchone() == (original_timeout, 1)
