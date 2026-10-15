"""The discovered B must reseed compatible generation B to reach C."""

from apps.knowledge_graph.projection.topology_adapter import (
    Neo4jProjectedTopologyQueryAdapter,
)
from apps.knowledge_graph.retrieval.topology.memgraph import (
    MemgraphProjectedTopologyLoader,
)
from apps.knowledge_graph.tests.frontier_fixture_support import (
    FrontierFixture,
    LegacyNeighborhoodDriver,
)


def test_actual_legacy_neighborhood_acquisition_misses_cross_generation_bridge():
    fixture = FrontierFixture()
    snapshot = MemgraphProjectedTopologyLoader(
        Neo4jProjectedTopologyQueryAdapter(LegacyNeighborhoodDriver(fixture.bundles))
    ).load(
        ready=fixture.group.scope.ready,
        seeds=fixture.seeds,
        caps=fixture.caps,
        deadline=fixture.deadline,
    )
    assert fixture.shared_identity in snapshot.identity_keys
    assert fixture.answer_identity not in snapshot.identity_keys


def test_discovered_identity_reseeds_other_generation(monkeypatch):
    from apps.knowledge_graph.projection.frontier_adapter_v2 import (
        Neo4jCanonicalFrontierAdapterV2,
    )
    from apps.knowledge_graph.retrieval.topology.frontier_loader_v2 import (
        canonical_frontier_loader_v2,
    )
    from apps.knowledge_graph.tests.frontier_index_fixture_driver import (
        FrontierIndexFixtureDriver,
    )

    fixture = FrontierFixture()
    from apps.knowledge_graph.projection.frontier_adapter_v2 import FrontierSessionV2

    calls = []
    original = FrontierSessionV2.prepare_copies

    def prepare(session, pairs, **kwargs):
        calls.extend(pairs)
        return original(session, pairs, **kwargs)

    monkeypatch.setattr(FrontierSessionV2, "prepare_copies", prepare)
    driver = FrontierIndexFixtureDriver(fixture.bundles)
    loader = canonical_frontier_loader_v2(
        selection=fixture.selection,
        codec=fixture.codec,
        adapter=Neo4jCanonicalFrontierAdapterV2(driver),
    )
    budget = fixture.budget()
    result = loader.load(
        group=fixture.group,
        seeds=fixture.seeds,
        request=fixture.request,
        caps=fixture.caps,
        budget=budget,
        deadline=fixture.deadline,
    )
    assert fixture.answer_identity in result.snapshot.identity_keys
    assert (fixture.generation_b, fixture.shared_identity) in calls
    for identity in {identity for _, identity in calls}:
        assert {g for g, i in calls if i == identity} == {
            b.generation.generation_key for b in fixture.bundles
        }
    assert len(calls) == len(set(calls))
    assert all(
        row.discovery_hop == 2
        for row in result.snapshot.audit_rows
        if row.kind == "automatic_membership"
        and row.automatic_membership_key == fixture.answer_identity
    )
    assert result.coverage.exhaustive is False
    assert budget.queries == len(driver.calls) <= 64
    assert all(
        c.representation_generation_key is None for c in result.evidence_candidates
    )
