"""An optional graph deadline must not strand captured ordinary source capacity."""

import pytest
from django.conf import settings

from apps.documents.models import TextChunk
from apps.documents.services.chunk_search_authorization import authorized_search_rows
from apps.documents.services.closed_search_finalizer import _bind_search_authorizer
from apps.documents.services.graph_source_admission_policy import charged_sources
from apps.documents.services.native_graph_search_closure import _try_native_search_closure
from apps.documents.tests.test_graph_source_admission import capture, pressure_rows
from apps.knowledge_graph.tests.test_graph_candidate_operation_work import graph_frame
from lib.retrieval.operation import operation_scope
from lib.retrieval.turn_budget import TurnBudget, TurnLimits

pytestmark = pytest.mark.django_db


def test_global_graph_deadline_completes_baseline_on_original_ledger(monkeypatch):
    from apps.documents.services import graph_candidate_admission, source_body_batch
    from apps.documents.services.source_deadline import _DEADLINE
    from apps.knowledge_graph.retrieval import ready_scope_repository_v2 as ready
    from apps.knowledge_graph.retrieval.ready_authority_reads_v2 import (
        capture_authority_check,
    )
    from lib import replay_observation

    monkeypatch.setenv("RAG_ACQUISITION_SCHEDULE", "native_closure")
    monkeypatch.setenv("RAG_SCOPED_BASELINE_RECOVERY_ENABLED", "0")
    monkeypatch.setattr(settings, "KG_OVERLAY_ENABLED", True, raising=False)
    monkeypatch.setattr(settings, "RAG_CANDIDATE_MULTIPLIER", 3.0, raising=False)
    monkeypatch.setattr(settings, "RAG_SHORT_QUERY_CANDIDATE_SCALE", 1.0, raising=False)
    now = [0.0]
    budget = TurnBudget(TurnLimits(), clock=lambda: now[0])
    with graph_frame(
        monkeypatch, prepare_sources=pressure_rows, dataset=True,
        budget=budget, timeout_ms=10_000,
    ) as active:
        fixture, graph, source, scope, _, dependencies, parent, _ = active
        graph.clock = lambda: now[0]
        # Two actual lexical channels plus a valid cap12 leave three baseline
        # slots when the follow-up starts with fifteen slots. No fake dense lane.
        dependencies.settings.graph_direct_max_candidates = 12
        dependencies.settings.graph_extended_max_candidates = 12
        assert len(scope.requested_document_ids) > 1
        assert scope.requested_document_ids == fixture.authorization.selected_document_ids

        assert budget.reserve_action("initial-global-search")
        with operation_scope(budget, timeout_ms=10_000, source_limit=30):
            first = capture(active)
            first.source_admission.top_up()
            assert budget.sources_used == 30
            assert budget.remaining_source_capacity() == 0
        first_charges = charged_sources(budget)
        assert len(first_charges) == 30 and budget.remaining_source_capacity() == 15

        assert budget.reserve_action("followup-global-search")
        with operation_scope(budget, timeout_ms=6500) as operation:
            snapshot = capture(active)
            plan = snapshot.source_admission
            assert plan.graph_ceiling == 12
            assert budget.sources_used == 33
            assert len(charged_sources(budget) - first_charges) == 3
            initial_ids = {row.pk for row in snapshot.baseline_candidates}
            assert len(initial_ids) == 33
            assert len(set(plan.references) - initial_ids) >= 12
            authorize = _bind_search_authorizer(
                authorized_search_rows, fixture.authorization, True,
            )
            original_deadlines = operation.deadline, parent.deadline
            before_text = budget.text_used.copy()
            expiries, fetched, graph_events = [], [], []
            select, fetch = ready._select_closed_ready_scope_v2, source_body_batch._fetch_body_rows

            def expire_ready_authority(**kwargs):
                select(**kwargs)  # Keep real persisted readiness/authority reads.
                check = capture_authority_check()
                bound = _DEADLINE.get()
                assert bound is not None and bound[2] is budget
                now[0] += bound[0] - bound[1]() + .001
                assert operation.active() and budget.can_publish()
                expiries.append(bound)
                check()  # The real child-deadline conversion must raise.
                pytest.fail("ready authority ignored its expired child deadline")

            def fetch_rows(query):
                rows = fetch(query)
                fetched.extend(row.pk for row in rows)
                return rows

            def unexpected_work(*args, **kwargs):
                pytest.fail("baseline completion reacquired ranks or requested embeddings")

            monkeypatch.setattr(ready, "_select_closed_ready_scope_v2", expire_ready_authority)
            monkeypatch.setattr(source_body_batch, "_fetch_body_rows", fetch_rows)
            monkeypatch.setattr(graph_candidate_admission, "capture_metadata", unexpected_work)
            monkeypatch.setattr(TextChunk, "get_chunk_embedding", unexpected_work)
            monkeypatch.setattr(replay_observation, "graph_selected_sources_v2", graph_events.append)
            result = _try_native_search_closure(
                snapshot, "Atlas", 15, authorize=authorize,
                dependencies=dependencies, overlay_enabled=True,
            )

            assert len(expiries) == 1 and result is not None
            assert result.graph_diagnostics["graph_status"] == "error"
            assert budget.sources_used == 45, "graph deadline stranded twelve ordinary source slots"
            rows = result.final_rows[2]
            ids = {row.pk for row in rows}
            assert len(ids) == 45 and initial_ids < ids
            assert len(ids - initial_ids) == 12
            assert ids - initial_ids <= set(fetched)
            assert ids <= set(plan.references)
            assert set(scope.bodies) == ids
            assert {row.pk for row in source.cache.values()} <= ids
            assert first_charges <= charged_sources(budget)
            assert budget.text_used["materialized"] > before_text["materialized"]
            assert budget.actions_used == 2
            assert budget.pairs_used == {"acquisition": 0, "final": 0}
            assert not graph_events and result.score_set is None
            assert (operation.deadline, parent.deadline) == original_deadlines
            assert operation.active() and parent.active()
