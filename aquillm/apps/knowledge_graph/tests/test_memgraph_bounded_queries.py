from __future__ import annotations

from dataclasses import replace

from apps.knowledge_graph.projection.serialization import projection_checksum
from apps.knowledge_graph.projection.topology_adapter import (
    Neo4jProjectedTopologyQueryAdapter,
    _selected_matches_bundle,
)
from apps.knowledge_graph.retrieval.topology.contracts import ProjectedSeedV1
from apps.knowledge_graph.retrieval.topology.memgraph import (
    MemgraphProjectedTopologyLoader,
)
from apps.knowledge_graph.tests.test_projected_topology_adapter import (
    ProjectionDriver,
    _caps,
    _ready,
)
from apps.knowledge_graph.tests.test_projection_records import _bundle


def test_native_adapter_maps_canonical_seed_and_bounds_all_families() -> None:
    bundle = _bundle()
    canonical_seed = bundle.automatic_memberships[1].automatic_membership_key
    assert canonical_seed is not None
    driver = ProjectionDriver(bundle)

    snapshot = MemgraphProjectedTopologyLoader(
        Neo4jProjectedTopologyQueryAdapter(driver, clock=lambda: 40.0)
    ).load(
        ready=_ready(bundle),
        seeds=(ProjectedSeedV1(canonical_seed, 1.0),),
        caps=_caps(),
        deadline=42.5,
    )

    family_calls = [
        call for call in driver.calls if "CollectionGeneration" not in call[0]
    ]
    assert canonical_seed in snapshot.identity_keys
    assert len(family_calls) == 9
    assert all("WITH DISTINCT n" in call[0] for call in family_calls)
    assert all("LIMIT $page_limit" in call[0] for call in family_calls)
    assert all(call[1]["seed_keys_csv"] == canonical_seed for call in family_calls)
    entity_query = next(
        call[0] for call in family_calls if "ProjectedEntity " in call[0]
    )
    assert "membership.automatic_membership_key" in entity_query
    assert "ENTITY_MEMBERSHIP" in entity_query
    assert all(
        "opaque_key:physical.relation_key" not in call[0]
        for call in family_calls
    )
    relationship_queries = [
        query
        for query, *_rest in family_calls
        if any(
            relationship in query
            for relationship in (
                "ENTITY_MEMBERSHIP",
                "PROJECTED_RELATION",
                "DOCUMENT_CHUNK",
                "RELATION_EVIDENCE",
                "ENTITY_MENTION",
            )
        )
    ]
    assert relationship_queries
    assert all("generation_key" in query for query in relationship_queries)
    path_queries = [
        call[0] for call in family_calls if "relationships(path)" in call[0]
    ]
    assert path_queries
    assert all(
        "edge.generation_key = $generation_key" in query for query in path_queries
    )


def test_bounded_subset_is_not_compared_to_whole_generation_checksum() -> None:
    original = _bundle()
    bounded = replace(
        original,
        relations=(),
        evidence=(),
        relation_semantics=(),
        counts=replace(
            original.counts,
            relation_count=0,
            evidence_count=0,
            relation_semantics_count=0,
        ),
    )

    full_checksum = _ready(original).selected_generations[0].graph_checksum
    assert projection_checksum(bounded) != full_checksum
    assert _selected_matches_bundle(_ready(original).selected_generations[0], bounded)
