"""The original native failure remains diagnosable without exposing source text."""

from contextlib import contextmanager
import json

import pytest

from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.knowledge_graph.tests.test_closed_graph_attempt_v2 import attempt_for
from apps.knowledge_graph.tests.test_graph_candidate_operation_work import graph_frame

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.mark.parametrize("failure,phase,kind,reason", [
    ("materialize", "materialization", "value_error", "unclassified"),
    ("source", "baseline_top_up", "source_scope_unavailable", "captured_source_changed"),
    ("sql", "pool_ordering", "database_error", "statement_cancelled"),
    ("cleanup", "database_cleanup", "runtime_error", "unclassified"),
    ("invalid", "materialization", "value_error", "unclassified"),
])
def test_native_failure_retains_actual_phase_and_safe_reason(monkeypatch, failure, phase, kind, reason):
    from django.db import connection
    from apps.knowledge_graph.retrieval import graph_candidate_operation_v2 as owner
    from apps.knowledge_graph.retrieval import ready_materialization_v2 as materializer

    with graph_frame(monkeypatch) as active:
        if failure in ("materialize", "invalid"):
            def fail(**kwargs):
                if failure == "invalid":
                    active[6].live = False
                raise ValueError("private query/document/provider detail must not be exported")
            monkeypatch.setattr(materializer, "_materialize_closed_rows", fail)
        elif failure == "source":
            def fail(*args):
                raise SourceScopeUnavailable("source_admission_captured_row_changed")
            monkeypatch.setattr(owner, "deferred_baseline", fail)
        elif failure == "sql":
            def fail(*args, **kwargs):
                with connection.cursor() as cursor:
                    cursor.execute("SET LOCAL statement_timeout = '1ms'")
                    cursor.execute("SELECT pg_sleep(0.05)")
                raise AssertionError("PostgreSQL failed to cancel")
            monkeypatch.setattr(owner, "order_graph_pool", fail)
        else:
            original, cleanups = owner.bounded_source_database, []
            @contextmanager
            def database(*args):
                with original(*args):
                    yield
                cleanups.append(True)
                if len(cleanups) == 2:
                    raise RuntimeError("private database cleanup detail")
            monkeypatch.setattr(owner, "bounded_source_database", database)
        outcome = attempt_for(active, active[4], "What is A?").run_native_closure(
            closure_reserve_ms=1250)
        assert outcome.kind == ("invalid" if failure == "invalid" else "failure")
        assert outcome.pool is None and active[2].cache == {} and active[3].bodies == {}
        assert active[2].budget.sources_used == 2
        assert "graph_failure_reason" not in outcome.diagnostics
        assert outcome.diagnostics.get("graph_failure") == {
            "version": "native-graph-failure-v1", "type": kind,
            "reason": reason, "phase": phase, "owner_valid": failure != "invalid",
        }
        assert "private" not in json.dumps(outcome.diagnostics)
        assert not connection.needs_rollback


def test_compatibility_fallback_keeps_the_original_safe_failure(monkeypatch):
    from apps.documents.services.hybrid_graph_v2 import hybrid_graph_candidate_pool_v2
    from apps.knowledge_graph.retrieval import ready_materialization_v2 as materializer

    with graph_frame(monkeypatch) as active:
        def fail(**kwargs):
            raise ValueError("private provider payload")
        monkeypatch.setattr(materializer, "_materialize_closed_rows", fail)
        rows, observation = hybrid_graph_candidate_pool_v2(
            active[4], "What is A?", active[0].authorization, active[5])
        assert tuple(row.pk for row in rows) == (active[0].chunks["A"].pk,)
        assert observation.get("graph_failure") == {
            "version": "native-graph-failure-v1", "type": "value_error",
            "reason": "unclassified", "phase": "materialization", "owner_valid": True,
        }
        assert "graph_failure_reason" not in observation
