"""A single-paper graph subdeadline is not the hard search deadline."""
from time import monotonic

import pytest

from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.tests._native_graph_search_closure_support import (
    native_case,
    run_current_or_native,
)
from apps.knowledge_graph.tests.test_closed_graph_limits_v2 import returned_envelope
from apps.knowledge_graph.tests.test_ready_scope import _DOC_A

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.mark.parametrize("boundary", ["graph", "action", "permission", "slice", "revoked_during_read", "committed", "default_off"])
def test_scoped_graph_deadline_only_recovers_fresh_captured_baseline(monkeypatch, boundary):
    from apps.documents.services import graph_candidate_admission, source_body_batch
    from lib import replay_observation

    gate = "RAG_SCOPED_BASELINE_RECOVERY_ENABLED"
    if boundary == "default_off":
        monkeypatch.delenv(gate, raising=False)
    else:
        monkeypatch.setenv(gate, "1")
    epoch = monotonic()
    with native_case(
        monkeypatch, fail=False, controlled_graph_clock=True, graph_epoch=epoch,
        requested=frozenset({_DOC_A}), timeout_ms=6500,
    ) as case:
        scope, budget = case.scope, case.source.budget
        initial = {r.pk for r in case.snapshot.baseline_candidates}
        plan = case.snapshot.source_admission
        assert scope.requested_document_ids == {_DOC_A} < case.f.authorization.selected_document_ids
        assert initial and set(plan.top_up_ids()) - initial
        original_action_deadline = case.operation.deadline
        after, fetched, published, read_cutoffs = [], [], [], []

        def expire(kw, envelope):
            after.append(kw["deadline"] - epoch)
            case.now[0] = (original_action_deadline if boundary == "action" else after[-1]) + .001
            if boundary == "permission":
                case.f.policy.rows = ()

        returned_envelope(monkeypatch, empty=True, reasons=("deadline",), after=expire)
        fetch = source_body_batch._fetch_body_rows

        def fetch_rows(query):
            rows = fetch(query)
            if after:
                from apps.documents.services.source_deadline import _DEADLINE
                deadline, clock, _ = _DEADLINE.get()
                read_cutoffs.append(case.now[0] + deadline - clock())
                fetched.extend(r.pk for r in rows)
                if boundary == "slice":
                    case.now[0] += deadline - clock() + .001
                elif boundary == "revoked_during_read":
                    case.f.policy.rows = ()
            return rows

        monkeypatch.setattr(source_body_batch, "_fetch_body_rows", fetch_rows)
        if boundary == "committed":
            from apps.knowledge_graph.retrieval.graph_candidate_stage_v2 import _GraphSourceStage
            publish = _GraphSourceStage.publish

            def expire_committed(stage, rows, check):
                result = publish(stage, rows, check)
                if after and read_cutoffs and set(scope.bodies) - initial:
                    case.now[0] = read_cutoffs[-1] + .001
                return result
            monkeypatch.setattr(_GraphSourceStage, "publish", expire_committed)
        monkeypatch.setattr(replay_observation, "graph_selected_sources_v2", published.append)
        monkeypatch.setattr(graph_candidate_admission, "capture_metadata",
                            lambda *a, **kw: pytest.fail("baseline ranks reacquired"))
        if boundary in {"graph", "committed"}:
            result = run_current_or_native(case)
            ids = {r.pk for r in result.final_rows[2]}
            assert initial < ids, "deferred authorized candidates were stranded"
            assert ids - initial <= set(fetched)
            assert ids == set(scope.bodies)
            assert result.graph_diagnostics["source_admission"]["fallback_top_up_count"] == len(ids - initial)
            assert {r.doc_id for r in result.final_rows[2]} == {_DOC_A}
            assert result.graph_diagnostics["graph_failure_reason"] == "deadline"
            assert not result.graph_diagnostics["source_admission"]["fallback_top_up_skipped"]
            assert result.schedule_version == "native-closure-v3"
        elif boundary == "default_off":
            result = run_current_or_native(case)
            assert {r.pk for r in result.final_rows[2]} == initial and not fetched
            assert after[0] == pytest.approx(5.0, abs=.01)  # Earlier 5s graph cap, no extra reserve.
            assert result.graph_diagnostics["source_admission"]["fallback_top_up_skipped"]
            assert result.schedule_version == "native-closure-v2"
        elif boundary == "slice":
            result = run_current_or_native(case)
            assert fetched and case.operation.active()
            assert {r.pk for r in result.final_rows[2]} == set(scope.bodies) == initial
        else:
            with pytest.raises(SourceScopeUnavailable):
                run_current_or_native(case)
            assert bool(fetched) is (boundary == "revoked_during_read")
            assert set(scope.bodies) == initial
        assert len(after) == 1 and not published and not case.inference
        assert budget.sources_used <= budget.limits.unique_sources == 45
        assert case.operation.deadline == original_action_deadline
