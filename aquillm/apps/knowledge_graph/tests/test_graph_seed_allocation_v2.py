"""Captured baseline prefix and identity allowance remain global across groups."""

from dataclasses import replace
from time import monotonic

import pytest

from apps.documents.services.chunk_search_scope import (
    FrozenRetrievalScope,
    bind_search_scope,
    reset_search_scope,
)
from apps.documents.services.source_loading import SourceRuntime, source_runtime_scope
from apps.documents.tests.hybrid_graph_test_support import selected_snapshot
from apps.knowledge_graph.retrieval.extended_seed_repository import (
    ExtendedSeedRepository,
)
from apps.knowledge_graph.retrieval.types import GraphExpansionSeed
from apps.knowledge_graph.tests.graph_runtime_fixture_v2 import persisted_runtime
from lib.retrieval.turn_budget import TurnBudget, TurnLimits

pytestmark = pytest.mark.django_db


@pytest.mark.parametrize("invalid", [None, "missing", "foreign", "stale"])
def test_actual_extended_seed_reads_share_global_prefix_and_budget(
    monkeypatch, invalid
):
    f, runtime, settings, driver = persisted_runtime(monkeypatch, dataset=True)
    settings.graph_direct_enabled = False
    settings.graph_extended_enabled = True
    settings.graph_extended_max_seeds = 2
    settings.graph_extended_timeout_ms = 4500
    runtime.projection_repository_factory = lambda: ExtendedSeedRepository(
        using="default"
    )
    rows = (f.chunks["A"], f.chunks["B"], f.answer)
    seeds = tuple(
        GraphExpansionSeed(r.pk, i, float(1 / i)) for i, r in enumerate(rows, 1)
    )
    baseline = replace(
        selected_snapshot(baseline=rows),
        documents=tuple(f.documents.values()),
        graph_seeds=seeds,
    )
    if invalid == "missing":
        baseline = None
    elif invalid == "foreign":
        baseline = replace(baseline, graph_seeds=(GraphExpansionSeed(999999, 1, 1.0),))
    elif invalid == "stale":
        from apps.documents.models import TextChunk

        TextChunk.objects.filter(pk=rows[0].pk).update(content="stale")
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
    token = bind_search_scope(scope, "Atlas", graph_enabled=True)
    try:
        with source_runtime_scope(source):
            result = runtime.run_v2(
                query="Atlas",
                authorization=f.authorization,
                representation_snapshot=f.snapshot,
                budget=budget,
                settings=settings,
                deadline=monotonic() + 5,
                baseline=baseline,
                requested_document_ids=scope.requested_document_ids,
            )
    finally:
        reset_search_scope(token)
    if invalid is None:
        assert len(result.selection.groups) == 2
        assert len({r.authority_group_checksum for r in result.handoff.candidates}) == 2
        assert "frontier_budget" in result.handoff.coverage.reasons
        assert result.frontier_budget.queries_used > len(driver.calls)
        assert result.frontier_budget.queries_used <= 64
        assert result.frontier_budget.accounting_complete
    else:
        assert not result.handoff.candidates
        assert "not_ready" in result.handoff.coverage.reasons
        assert not driver.calls


@pytest.mark.parametrize("group_count", [2, 3])
def test_global_identity_cap_cannot_multiply_per_group(monkeypatch, group_count):
    from types import SimpleNamespace

    from apps.knowledge_graph.retrieval import graph_seed_allocation_v2 as allocation
    from apps.knowledge_graph.retrieval.topology.contracts import ProjectedSeedV1
    from apps.knowledge_graph.tests.frontier_fixture_support import FrontierFixture

    groups = tuple(SimpleNamespace(group_checksum=str(i)) for i in range(group_count))
    monkeypatch.setattr(
        allocation,
        "group_seeds",
        lambda **kw: (
            tuple(
                ProjectedSeedV1(
                    str(int(kw["group"].group_checksum) * 3 + j) * 64, 1 / 3
                )
                for j in range(1, 4)
            ),
            set(),
        ),
    )
    result, reasons = allocation._allocate(
        groups=groups,
        runtime=None,
        query="x",
        baseline=None,
        authorization=None,
        settings=SimpleNamespace(graph_extended_enabled=False),
        caps=SimpleNamespace(max_seeds=2),
        deadline=100,
        frontier=FrontierFixture().budget(),
    )
    assert sum(len(row[0]) for row in result.values()) == 2
    assert sum(bool(row[0]) for row in result.values()) == 2
    assert all(len(row[0]) <= 1 for row in result.values())
    assert sum(row[1] for row in result.values()) == 1.0
    assert reasons == {"frontier_budget"}
