"""Genuine native eligibility selects and observes the new acquisition version."""

import pytest

from apps.collections.tests.native_closed_seed_policy_fixtures import small_native_scope
from apps.documents.tests.test_graph_source_admission import capture
from apps.knowledge_graph.retrieval.frontier_ranking_validation_v2 import (
    CONSUMED_POLICY,
)
from apps.knowledge_graph.tests.test_graph_candidate_operation_work import graph_frame

pytestmark = pytest.mark.django_db(transaction=True, databases="__all__")


@pytest.mark.parametrize("fail", [False, True])
def test_native_owner_selects_new_loader_and_final_observation_even_on_failure(
    monkeypatch, fail
):
    from apps.documents.services.hybrid_graph_v2 import hybrid_graph_candidate_pool_v2
    from apps.knowledge_graph.projection.frontier_adapter_v2 import (
        Neo4jCanonicalFrontierAdapterV2,
    )
    from apps.knowledge_graph.retrieval import (
        graph_frontier_observation_v2 as observation,
    )
    from apps.knowledge_graph.retrieval import graph_runtime_execution_v2 as execution

    with graph_frame(
        monkeypatch, prepare_sources=small_native_scope, dataset=True
    ) as active:
        f, graph, _, scope, _, dependencies, _, _ = active
        graph.authorization = scope.authorization
        del graph._direct_seeds
        from apps.knowledge_graph.tests.test_native_ontology_proof_postgres import (
            extractor_runtime,
        )

        extractor_runtime(graph)
        dependencies.settings.graph_direct_enabled = False
        dependencies.settings.graph_extended_enabled = True
        dependencies.settings.graph_extended_timeout_ms = 5000
        snapshot = capture(active)
        policies, observations, handoffs = [], [], []
        original_open, original_capture = (
            Neo4jCanonicalFrontierAdapterV2.open,
            observation.capture_frontier_budget_v2,
        )

        def open_session(adapter, **kwargs):
            policies.append(adapter.acquisition_policy)
            session = original_open(adapter, **kwargs)
            if fail:

                def reject():
                    raise TimeoutError("fixture deadline after native policy selection")

                session.initialize = reject
            return session

        def observed(budget, **kwargs):
            observations.append(kwargs["acquisition_policy"])
            return original_capture(budget, **kwargs)

        original_select = execution.select_graph_evidence_v2

        def select(**kwargs):
            result = original_select(**kwargs)
            handoffs.append(result)
            return result

        monkeypatch.setattr(Neo4jCanonicalFrontierAdapterV2, "open", open_session)
        monkeypatch.setattr(observation, "capture_frontier_budget_v2", observed)
        monkeypatch.setattr(execution, "select_graph_evidence_v2", select)
        rows, diagnostics = hybrid_graph_candidate_pool_v2(
            snapshot, "Atlas", f.authorization, dependencies
        )
        assert policies and set(policies) == {CONSUMED_POLICY}
        assert observations == [CONSUMED_POLICY]
        if fail:
            assert f.chunks["B"].pk not in {row.pk for row in rows}
        else:
            assert handoffs and handoffs[0].candidates
            assert diagnostics["graph_status"] in {"hit", "miss"}
            assert f.chunks["A"].pk in {row.pk for row in rows}


def test_public_route_preserves_legacy_policy_and_observation(monkeypatch):
    from apps.knowledge_graph.retrieval import (
        graph_frontier_observation_v2 as observation,
    )
    from apps.knowledge_graph.retrieval.frontier_ranking_validation_v2 import POLICY
    from apps.knowledge_graph.tests.test_graph_runtime_v2 import setup_runtime
    from lib.retrieval.turn_budget import TurnBudget, TurnLimits

    m, f, runtime, settings, _, _ = setup_runtime(monkeypatch)
    observed = []
    original = observation.capture_frontier_budget_v2

    def capture_budget(budget, **kwargs):
        observed.append(kwargs["acquisition_policy"])
        return original(budget, **kwargs)

    monkeypatch.setattr(observation, "capture_frontier_budget_v2", capture_budget)
    m.run_graph_v2(
        query="query",
        runtime=runtime,
        authorization=runtime.authorization,
        representation_snapshot=f.selection.representation_snapshot,
        budget=TurnBudget(TurnLimits()),
        settings=settings,
        deadline=f.deadline,
    )
    assert observed == [POLICY]


@pytest.mark.parametrize(
    "change",
    ["permission", "body", "source", "representation", "generation", "cleanup"],
)
def test_native_consumed_evidence_keeps_fresh_publication_fences(monkeypatch, change):
    from contextlib import contextmanager
    from uuid import uuid4

    from apps.collections.models import CollectionPermission
    from apps.documents.models import (
        DocumentRepresentationState,
        RawTextDocument,
        TextChunk,
    )
    from apps.documents.services.hybrid_graph_v2 import hybrid_graph_candidate_pool_v2
    from apps.knowledge_graph.models import CollectionGraphProjection
    from apps.knowledge_graph.retrieval import graph_candidate_operation_v2 as operation
    from apps.knowledge_graph.tests.test_graph_candidate_operation_races import (
        after_hydration,
    )
    from apps.knowledge_graph.tests.test_native_ontology_proof_postgres import (
        extractor_runtime,
    )

    with graph_frame(
        monkeypatch, prepare_sources=small_native_scope, dataset=True
    ) as active:
        f, graph, source, scope, _, dependencies, _, _ = active
        graph.authorization = scope.authorization
        del graph._direct_seeds
        extractor_runtime(graph)
        dependencies.settings.graph_direct_enabled = False
        dependencies.settings.graph_extended_enabled = True
        dependencies.settings.graph_extended_timeout_ms = 5000
        snapshot = capture(active)
        original_lexical_ids = set(snapshot.trigram_chunk_ids)
        observed = []

        def mutate():
            observed.append(True)
            if change == "permission":
                CollectionPermission.objects.filter(
                    collection_id=7, permission="VIEW"
                ).delete()
            elif change == "body":
                TextChunk.objects.filter(pk=f.chunks["A"].pk).update(content="changed")
            elif change == "source":
                RawTextDocument.objects.filter(pk=f.documents["A"].pk).update(
                    full_text_hash="f" * 64
                )
            elif change == "representation":
                DocumentRepresentationState.objects.filter(
                    document_id=f.documents["A"].id
                ).update(legacy_closed=True)
            elif change == "generation":
                CollectionGraphProjection.objects.filter(
                    pk=f.projections["A"].pk
                ).update(generation_key=uuid4())

        after_hydration(monkeypatch, mutate)
        if change == "cleanup":
            original = operation.source_preparation_scope

            @contextmanager
            def preparation(*args):
                with original(*args):
                    yield
                raise ValueError("fixture preparation cleanup failure")

            monkeypatch.setattr(operation, "source_preparation_scope", preparation)
        rows, diagnostics = hybrid_graph_candidate_pool_v2(
            snapshot, "Atlas", f.authorization, dependencies
        )
        assert observed == [True]
        assert diagnostics["graph_status"] in {"error", "miss"}
        assert {row.pk for row in rows} <= original_lexical_ids
        assert {row.pk for row in source.cache.values()} <= original_lexical_ids
        assert set(scope.bodies) <= original_lexical_ids
        assert f.chunks["B"].pk not in {row.pk for row in rows}
        assert f.chunks["B"].pk not in {row.pk for row in source.cache.values()}
        assert f.chunks["B"].pk not in scope.bodies
        if change == "body":
            assert all(row.content != "changed" for row in rows)
