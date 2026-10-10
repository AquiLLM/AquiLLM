"""Actual early lifecycle and later pool dispatch never double-run graph work."""

from dataclasses import replace

import pytest

from apps.documents.services.chunk_search_scope import (
    FrozenRetrievalScope,
    bind_search_scope,
    reset_search_scope,
)
from apps.documents.services.hybrid_graph_authorization import (
    HybridGraphRetrievalDependencies,
)
from apps.documents.services.hybrid_graph_lifecycle import start_graph
from apps.documents.services.hybrid_graph_orchestration import (
    hybrid_graph_candidate_pool,
)
from apps.documents.services.source_loading import SourceRuntime, source_runtime_scope
from apps.documents.tests.hybrid_graph_test_support import selected_snapshot
from apps.knowledge_graph.tests.graph_runtime_fixture_v2 import persisted_runtime
from lib.retrieval.turn_budget import TurnBudget, TurnLimits

pytestmark = pytest.mark.django_db


@pytest.mark.parametrize("mode", ["v2", "shadow", "wrong", None])
def test_nonlegacy_mode_never_prestarts_legacy_scheduler(monkeypatch, mode):
    f, runtime, settings, _ = persisted_runtime(monkeypatch)
    settings.memgraph_traversal_enabled = True
    settings.graph_remediation_mode = mode
    attempts = []
    monkeypatch.setattr(
        "apps.knowledge_graph.retrieval.scheduler.HybridGraphBranchScheduler.start",
        lambda *args, **kwargs: attempts.append(kwargs),
    )
    dependencies = HybridGraphRetrievalDependencies(
        runtime, settings, runtime.materialize
    )
    assert start_graph("A", f.authorization, dependencies, True) is None
    assert attempts == []


def test_legacy_mode_keeps_existing_early_dispatch(monkeypatch):
    f, runtime, settings, _ = persisted_runtime(monkeypatch)
    settings.memgraph_traversal_enabled = True
    settings.graph_remediation_mode = "legacy"
    sentinel, attempts = object(), []

    def started(*args, **kwargs):
        attempts.append(kwargs)
        return sentinel

    monkeypatch.setattr(
        "apps.knowledge_graph.retrieval.scheduler.HybridGraphBranchScheduler.start",
        started,
    )
    dependencies = HybridGraphRetrievalDependencies(
        runtime, settings, runtime.materialize
    )
    assert start_graph("A", f.authorization, dependencies, True) is sentinel
    assert len(attempts) == 1


@pytest.mark.parametrize("race", [None, "body", "ready", "revoke"])
def test_later_production_dispatch_reuses_original_scope_and_final_fences(
    monkeypatch, race
):
    from apps.documents.models import TextChunk
    from apps.knowledge_graph.models import CollectionGraphProjection
    from uuid import uuid4

    f, runtime, settings, driver = persisted_runtime(monkeypatch)
    settings.memgraph_traversal_enabled = True
    budget = TurnBudget(TurnLimits())
    source = SourceRuntime(budget, f.authorization)
    scope = FrozenRetrievalScope(
        f.snapshot,
        None,
        None,
        f.authorization,
        "default",
        source,
        requested_document_ids=f.authorization.selected_document_ids,
    )
    snapshot = replace(
        selected_snapshot(baseline=()), documents=tuple(f.documents.values())
    )
    calls = []
    run = runtime.run_v2

    def execute(**kwargs):
        calls.append(kwargs)
        result = run(**kwargs)
        if race == "body":
            TextChunk.objects.filter(pk=f.chunks["A"].pk).update(content="changed")
        elif race == "ready":
            CollectionGraphProjection.objects.filter(pk=f.projections["A"].pk).update(
                generation_key=uuid4()
            )
        elif race == "revoke":
            f.policy.rows = ()
        return result

    runtime.run_v2 = execute
    dependencies = HybridGraphRetrievalDependencies(
        runtime, settings, runtime.materialize
    )
    token = bind_search_scope(scope, "What is A?", graph_enabled=True)
    try:
        with source_runtime_scope(source):
            handle = start_graph("What is A?", f.authorization, dependencies, True)
            rows, diagnostics = hybrid_graph_candidate_pool(
                snapshot, "What is A?", f.authorization, dependencies, handle=handle
            )
    finally:
        reset_search_scope(token)
    assert len(calls) == 1
    assert calls[0]["budget"] is budget
    assert calls[0]["representation_snapshot"] is f.snapshot
    assert bool(rows) is (race is None)
    assert len(driver.calls) <= 64
    if race is None:
        assert {r.doc_id for r in rows} == set(f.authorization.selected_document_ids)
        assert diagnostics["graph_coverage_v2"]["exhaustive"] is False
        assert budget.sources_used == 2
        assert budget.text_used["materialized"] == 2
