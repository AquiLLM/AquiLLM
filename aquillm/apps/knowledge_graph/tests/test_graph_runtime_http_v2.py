"""Actual V2 factory and HTTP path reach query selection and original body ledger."""

from time import monotonic

import pytest

from apps.documents.services.source_loading import SourceRuntime, source_runtime_scope
from apps.knowledge_graph.projection.frontier_cypher_v2 import (
    SCOPE_RAW_COMPLETE_BATCH_LOOKUP,
)
from apps.knowledge_graph.retrieval import graph_runtime_execution_v2 as graph
from apps.knowledge_graph.retrieval import graph_runtime_v2 as graph_runtime
from apps.knowledge_graph.retrieval.topology.factory import (
    create_canonical_frontier_loader_v2,
)
from apps.knowledge_graph.tests.graph_http_fixture_v2 import graph_http_transport
from apps.knowledge_graph.tests.graph_runtime_fixture_v2 import persisted_runtime
from lib.retrieval.turn_budget import TurnBudget, TurnLimits

pytestmark = pytest.mark.django_db


def test_actual_whole_runtime_http_factory_selection_and_body_budget(monkeypatch):
    f, runtime, settings, driver = persisted_runtime(monkeypatch, dataset=True)
    frontiers = []
    original_budget = graph_runtime.FrontierBudgetV2

    def capture_budget(*args, **kwargs):
        frontier = original_budget(*args, **kwargs)
        frontiers.append(frontier)
        return frontier

    monkeypatch.setattr(graph_runtime, "FrontierBudgetV2", capture_budget)
    prior_queries = []

    def capture_loader(**kwargs):
        prior_queries.append(frontiers[0].queries)
        return create_canonical_frontier_loader_v2(**kwargs)

    monkeypatch.setattr(
        graph,
        "create_canonical_frontier_loader_v2",
        capture_loader,
    )
    budget = TurnBudget(TurnLimits())
    with graph_http_transport(monkeypatch, driver) as (gateway, calls):
        runtime.topology_loader.driver = gateway
        with source_runtime_scope(SourceRuntime(budget, f.authorization)):
            result = runtime.run_v2(
                query="Which dataset does Atlas use?",
                authorization=f.authorization,
                representation_snapshot=f.snapshot,
                budget=budget,
                settings=settings,
                deadline=monotonic() + 5,
                requested_document_ids=frozenset({f.documents["A"].id}),
            )
    assert result.handoff.candidates
    assert len(result.handoff.candidates) == 2
    assert budget.sources_used == 3
    assert len(frontiers) == 1
    assert len(prior_queries) == 1
    assert any(query == SCOPE_RAW_COMPLETE_BATCH_LOOKUP for query, _ in driver.calls)
    assert 0 < len(calls) < len(driver.calls) <= 64
    assert frontiers[0].queries == prior_queries[0] + len(driver.calls) <= 64
    assert 0 < frontiers[0].wire_bytes == sum(size for size, _ in calls) <= 1048576
    assert all(0 < timeout <= 4500 for _, timeout in calls)
    print(
        "G3_HTTP_WHOLE_RUNTIME",
        len(calls),
        sum(size for size, _ in calls),
        budget.sources_used,
        budget.text_used["materialized"],
    )
