"""Actual early lifecycle and later pool dispatch never double-run graph work."""

from dataclasses import replace

import pytest

from apps.documents.services.chunk_search_scope import (
    freeze_search_scope,
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
    from uuid import uuid4

    from apps.documents.models import TextChunk
    from apps.knowledge_graph.models import CollectionGraphProjection

    f, runtime, settings, driver = persisted_runtime(monkeypatch)
    settings.memgraph_traversal_enabled = True
    budget = TurnBudget(TurnLimits())
    source = SourceRuntime(budget, f.authorization)
    with source_runtime_scope(source):
        scope = freeze_search_scope(
            tuple(f.documents.values()), using="default", authorization=f.authorization,
        )
    f.snapshot = scope.representation
    assert source._frozen_retrieval_scope.representation is scope.representation
    snapshot = replace(
        selected_snapshot(baseline=()),
        documents=tuple(f.documents.values()),
        baseline_candidates=(),
        vector_results=(),
        vector_chunk_ids=(),
        graph_seeds=(),
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


@pytest.mark.parametrize("narrow", [False, True])
@pytest.mark.parametrize("rrf", [False, True])
def test_actual_seed_extractor_runs_once_after_baseline_under_original_deadline(
    monkeypatch,
    narrow,
    rrf,
):
    from time import perf_counter
    from types import SimpleNamespace

    from lib.knowledge_graph.query_extractor.contracts import (
        QueryEntitySpanV1,
        QueryExtractionResponseV1,
    )
    from lib.knowledge_graph.tests.test_query_extractor_contracts import _provenance

    monkeypatch.setattr(
        "apps.documents.services.chunk_search_rank_policy.controls",
        lambda: {"rrf": rrf, "protected_exact": rrf, "full_text": False},
    )

    f, runtime, settings, driver = persisted_runtime(monkeypatch)
    incoming = f.authorization
    if narrow:
        from apps.collections.services.retrieval_authorization import (
            freeze_retrieval_authorization_context,
        )

        incoming = freeze_retrieval_authorization_context(
            principal=f.authorization.reauthorization_capability._principal,
            database_alias="default",
            policy=f.policy,
            selected_collection_ids=(f.documents["A"].collection_id,),
            selected_document_ids=(f.documents["A"].id,),
            reauthorization_capability=f.authorization.reauthorization_capability,
        )
        runtime.authorization = incoming
    settings.memgraph_traversal_enabled = True
    settings.graph_overall_timeout_ms = 4500
    del runtime._direct_seeds
    attempts = []

    def extractor(**kwargs):
        def extract(**call):
            attempts.append(call)
            return QueryExtractionResponseV1(
                replace(_provenance(), ontology_checksum=kwargs["ontology"].checksum),
                1,
                1,
                (QueryEntitySpanV1("model", 0, 1, 0.99),),
            )

        return SimpleNamespace(extract=extract)

    runtime.extractor_factory = extractor
    dispatched = []
    original_run = type(runtime).run_v2

    def run(self, **kwargs):
        dispatched.append(kwargs)
        return original_run(self, **kwargs)

    monkeypatch.setattr(type(runtime), "run_v2", run)
    budget = TurnBudget(TurnLimits())
    source = SourceRuntime(budget, f.authorization)
    with source_runtime_scope(source):
        scope = freeze_search_scope(
            (f.documents["A"],), using="default", authorization=incoming,
        )
    f.snapshot = scope.representation
    assert source._frozen_retrieval_scope.representation is scope.representation
    snapshot = replace(
        selected_snapshot(baseline=()),
        documents=(f.documents["A"],),
        baseline_candidates=(),
        vector_results=(),
        vector_chunk_ids=(),
        graph_seeds=(),
    )
    deps = HybridGraphRetrievalDependencies(runtime, settings, runtime.materialize)
    token = bind_search_scope(scope, "A", graph_enabled=True)
    started = perf_counter()
    try:
        with source_runtime_scope(source):
            from apps.documents.models import TextChunk
            from apps.documents.services.source_loading import source_query_rows

            assert source_query_rows(TextChunk.objects.filter(pk=f.chunks["A"].pk))
            assert budget.sources_used == 1
            handle = start_graph("A", incoming, deps, True)
            assert not attempts
            rows, diagnostics = hybrid_graph_candidate_pool(
                snapshot, "A", incoming, deps, handle=handle
            )
    finally:
        reset_search_scope(token)
    assert len(attempts) == 1
    assert len(dispatched) == 1 and dispatched[0]["budget"] is budget
    assert attempts[0]["deadline"] <= dispatched[0]["deadline"] <= budget._deadline
    assert perf_counter() - started < 4.5
    assert rows and {r.doc_id for r in rows} == {f.documents["A"].id}
    assert diagnostics["frontier_budget"]["queries_used"] > len(driver.calls)
    assert diagnostics["frontier_budget"]["accounting_complete"]
    assert budget.sources_used == 1
    if rrf:
        assert len(scope.fusion_v2.candidates) == 1
        ranks = scope.fusion_v2.candidates[0].channel_ranks
        assert tuple((r.channel, r.rank) for r in ranks) == (("graph_v2", 1),)
        assert scope.fusion_v2.candidates[0].rrf_score == 1 / 61
