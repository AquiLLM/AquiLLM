from __future__ import annotations

from dataclasses import replace
from hashlib import sha256

from apps.knowledge_graph.projection.topology_adapter import (
    Neo4jProjectedTopologyQueryAdapter,
)
from apps.knowledge_graph.retrieval.topology.contracts import (
    HybridBranchKind,
    ProjectedSeedV1,
    TopologyCapsV1,
)
from apps.knowledge_graph.retrieval.topology.memgraph import (
    MemgraphProjectedTopologyLoader,
)
from apps.knowledge_graph.tests.test_projected_topology_adapter import (
    ProjectionDriver,
    _ready,
)
from apps.knowledge_graph.tests.test_projection_records import _bundle


def _key(label: str) -> str:
    return sha256(label.encode()).hexdigest()


def _fanout_bundle():
    original = _bundle()
    source = original.entities[0]
    source_membership = original.automatic_memberships[0]
    entities = [source]
    memberships = [source_membership]
    relations, evidence = [], []
    for index in range(11):
        entity_key = _key(f"fanout-entity-{index}")
        target = replace(
            original.entities[1],
            entity_key=entity_key,
            cluster_key=_key(f"fanout-cluster-{index}"),
        )
        membership = replace(
            original.automatic_memberships[1],
            entity_key=entity_key,
            automatic_membership_key=None,
        )
        relation_key = _key(f"fanout-relation-{index}")
        relation = replace(
            original.relations[0],
            relation_key=relation_key,
            target_entity_key=entity_key,
        )
        evidence_key = _key(f"fanout-evidence-{index}")
        evidence_row = replace(
            original.evidence[0],
            evidence_key=evidence_key,
            relation_key=relation_key,
            relation_mention_key=_key(f"fanout-relation-mention-{index}"),
            head_mention_key=_key(f"fanout-head-{index}"),
            tail_mention_key=_key(f"fanout-tail-{index}"),
            tail_mapping_key=_key(f"fanout-tail-mapping-{index}"),
            provenance_key=_key(f"fanout-provenance-{index}"),
            semantic_signature=_key(f"fanout-signature-{index}"),
        )
        entities.append(target)
        memberships.append(membership)
        relations.append(relation)
        evidence.append(evidence_row)
    entities.sort(key=lambda row: row.entity_key)
    memberships.sort(key=lambda row: row.entity_key)
    relations.sort(key=lambda row: row.relation_key)
    evidence.sort(key=lambda row: row.evidence_key)
    counts = replace(
        original.counts,
        entity_count=12,
        automatic_membership_count=12,
        relation_count=11,
        evidence_count=11,
    )
    return replace(
        original,
        entities=tuple(entities),
        automatic_memberships=tuple(memberships),
        relations=tuple(relations),
        evidence=tuple(evidence),
        counts=counts,
    )


def test_adapter_enforces_signed_max_fanout_per_source() -> None:
    bundle = _fanout_bundle()
    source = next(
        row.entity_key
        for row in bundle.entities
        if row.entity_key == _bundle().entities[0].entity_key
    )
    snapshot = MemgraphProjectedTopologyLoader(
        Neo4jProjectedTopologyQueryAdapter(ProjectionDriver(bundle), clock=lambda: 40.0)
    ).load(
        ready=_ready(bundle),
        seeds=(ProjectedSeedV1(source, 1.0),),
        caps=TopologyCapsV1(HybridBranchKind.DIRECT, 32, 1, 200, 1_000, 20),
        deadline=42.5,
    )

    assert (
        sum(row.source_identity_key == source for row in snapshot.relation_groups) == 10
    )
