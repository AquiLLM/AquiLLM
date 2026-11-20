"""An initial action keeps native admission while reserving novel follow-up rows."""

import pytest
from django.conf import settings

from apps.documents.models import TextChunk
from apps.documents.services.chunk_search_authorization import authorized_search_rows
from apps.documents.services.closed_search_finalizer import _bind_search_authorizer
from apps.documents.services.graph_source_admission_policy import charged_sources
from apps.documents.services.native_graph_search_closure import (
    _try_native_search_closure,
)
from apps.documents.services.native_search_capture import capture_search
from apps.documents.services.source_loading import source_query_rows
from apps.documents.tests.test_graph_source_admission import capture, pressure_rows
from apps.knowledge_graph.tests.test_graph_candidate_operation_work import graph_frame
from lib.retrieval.operation import operation_scope

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.fixture(autouse=True)
def native_schedule(monkeypatch):
    monkeypatch.setenv("RAG_ACQUISITION_SCHEDULE", "native_closure")
    monkeypatch.setattr(settings, "RAG_CANDIDATE_MULTIPLIER", 3.0, raising=False)
    monkeypatch.setattr(settings, "RAG_SHORT_QUERY_CANDIDATE_SCALE", 1.0, raising=False)


@pytest.mark.parametrize("graph_on", [False, True])
def test_initial_source_reserve_preserves_native_plan_and_fifteen_followup_slots(
    monkeypatch, graph_on,
):
    monkeypatch.setattr(settings, "KG_OVERLAY_ENABLED", graph_on, raising=False)
    with graph_frame(monkeypatch, prepare_sources=pressure_rows, dataset=True) as active:
        fixture, _, source, scope, _, dependencies, _, _ = active
        budget = source.budget
        scope.graph_enabled = graph_on
        with operation_scope(budget, timeout_ms=10_000, source_limit=30):
            snapshot = capture(active, native=graph_on)
            expected_initial = 10 if graph_on else 30
            assert budget.sources_used == expected_initial
            assert len(snapshot.baseline_candidates) == expected_initial
            authorize = _bind_search_authorizer(
                authorized_search_rows, fixture.authorization, graph_on,
            )
            assert capture_search(
                snapshot, "Atlas", 15, authorize,
                dependencies if graph_on else None, graph_on,
            ) is not None
            if graph_on:
                plan = snapshot.source_admission
                assert set(plan.initial_ids) == {
                    row.pk for row in snapshot.baseline_candidates
                }
                assert plan.graph_ceiling == 20
                assert len(plan.top_up_ids()) == 20
                assert len(plan.top_up()) == 20
            assert budget.sources_used == 30
            assert budget.remaining_source_capacity() == 0

        before = charged_sources(budget)
        assert budget.remaining_source_capacity() == 15
        with operation_scope(budget, timeout_ms=10_000):
            rows = source_query_rows(
                TextChunk.objects.filter(doc_id=fixture.chunks["A"].doc_id).order_by("pk"),
                runtime=source,
            )
            assert len(rows) == 45
            assert len(charged_sources(budget) - before) == 15
            assert budget.sources_used == budget.limits.unique_sources == 45
            assert budget.remaining_source_capacity() == 0


def test_native_graph_expansion_and_top_up_share_initial_action_ceiling(monkeypatch):
    from hashlib import md5

    from apps.knowledge_graph.retrieval import graph_runtime_execution_v2 as execution

    def admission_identity(row):
        return row.pk, md5(row.content.encode("utf-8"), usedforsecurity=False).hexdigest()

    selections = []
    select = execution.select_graph_evidence_v2

    def observe_selection(**kwargs):
        handoff = select(**kwargs)
        selections.append((kwargs["envelope"].inputs, handoff.candidates))
        assert kwargs["max_mentions_per_identity"] == 2
        return handoff

    monkeypatch.setattr(execution, "select_graph_evidence_v2", observe_selection)
    monkeypatch.setattr(settings, "KG_OVERLAY_ENABLED", True, raising=False)
    with graph_frame(monkeypatch, prepare_sources=pressure_rows, dataset=True) as active:
        fixture, _, source, scope, _, dependencies, _, _ = active
        budget = source.budget
        with operation_scope(budget, timeout_ms=10_000, source_limit=30):
            snapshot = capture(active)
            assert budget.sources_used == 10
            result = _try_native_search_closure(
                snapshot, "Atlas", 15,
                authorize=_bind_search_authorizer(
                    authorized_search_rows, fixture.authorization, True,
                ),
                dependencies=dependencies,
                overlay_enabled=True,
            )
            assert result is not None
            assert result.graph_diagnostics["graph_status"] == "hit"
            assert fixture.chunks["B"].pk in {row.pk for row in result.hybrid_pool}
            assert budget.sources_used == 30
            # The fixture has three Atlas mentions. For query "Atlas", the two
            # higher-confidence model mentions fill the per-identity selector
            # allowance. The Cedar mention was read, so its charge remains even
            # though neither the graph selector nor baseline ranks retain it.
            assert len(selections) == 1
            hydrated, selected = selections[0]
            assert len(hydrated) == 4 and len(selected) == 3
            rejected = tuple(row for row in hydrated if row.text == fixture.answer.content)
            assert len(rejected) == 1
            assert rejected[0].ref.chunk_key not in {row.chunk_key for row in selected}
            returned = {admission_identity(row) for row in result.hybrid_pool}
            assert charged_sources(budget) - returned == {
                admission_identity(fixture.answer)
            }
            assert len(result.hybrid_pool) == 29
            assert fixture.answer.pk not in scope.bodies
            assert fixture.answer.pk not in {row.pk for row in source.cache.values()}
            assert set(scope.bodies) >= {row.pk for row in result.hybrid_pool}
            assert budget.remaining_source_capacity() == 0
        assert budget.remaining_source_capacity() == 15
