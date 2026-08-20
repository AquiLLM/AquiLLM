from __future__ import annotations

import json
from hashlib import sha256

import pytest

from apps.knowledge_graph.retrieval import projected_types as t
from apps.knowledge_graph.retrieval.topology.contracts import (
    HybridBranchKind,
    ProjectedSeedV1,
    TopologyCapsV1,
    TopologyQueryName,
)
from apps.knowledge_graph.retrieval.topology.memgraph import (
    MemgraphProjectedTopologyLoader,
    TopologyLoadError,
)
from apps.knowledge_graph.tests.test_memgraph_topology import Driver, K, ready, snapshot


def _key(label: str) -> str:
    return sha256(label.encode()).hexdigest()


def graph_snapshot():
    source, target = K[4], _key("target")
    target_entity, relation = _key("target-entity"), _key("relation")
    evidence_key, chunk_key = _key("evidence"), _key("chunk")
    evidence = t.ProjectedChunkEvidenceV1(chunk_key, K[2], 2, 0.75, evidence_key)
    signature = t.ProjectedEvidenceSignatureV1(
        evidence_key,
        relation,
        _key("relation-mention"),
        chunk_key,
        K[2],
        2,
        0.75,
        K[6],
        K[2],
        _key("head"),
        _key("tail"),
        "related_to",
        _key("head-map"),
        _key("tail-map"),
        t.ProjectedEvidenceOrientationV1.HEAD_TO_TAIL,
        K[11],
        K[14],
    )
    audits = (
        t.ProjectedAutomaticMembershipAuditV1(0, K[3], source, K[9], "resolver-v1"),
        t.ProjectedAutomaticMembershipAuditV1(
            0, target_entity, target, _key("target-decision"), "resolver-v1"
        ),
        t.ProjectedPhysicalRelationAuditV1(
            1, relation, K[5], K[3], "related_to", target_entity
        ),
        t.ProjectedRelationEvidenceAuditV1(1, signature),
    )
    return t.ProjectedAuthorizedGraphSnapshotV1(
        snapshot().algorithm,
        snapshot().caps,
        2,
        snapshot().allowed_scope,
        tuple(sorted((source, target))),
        (),
        (
            t.ProjectedRelationGroupV1(
                source,
                "related_to",
                target,
                t.ProjectedRetrievalDirectionV1.FORWARD,
                1.0,
                1,
                (evidence,),
            ),
        ),
        (),
        snapshot().artifact_provenance,
        tuple(
            sorted(
                audits, key=lambda row: (row.discovery_hop, row.kind, t._canonical(row))
            )
        ),
    )


def _load(driver):
    return MemgraphProjectedTopologyLoader(driver).load(
        ready=ready(),
        seeds=(ProjectedSeedV1(K[4], 1.0),),
        caps=TopologyCapsV1(HybridBranchKind.DIRECT, 32, 2, 200, 1_000, 20),
        deadline=42.5,
    )


def test_memgraph_loader_constructs_graph_from_all_response_families() -> None:
    graph = graph_snapshot()
    assert _load(Driver(graph)) == graph


@pytest.mark.parametrize(
    ("query", "section_json"),
    (
        (
            TopologyQueryName.RELATION_TOPOLOGY,
            json.dumps({"relation_groups": []}, separators=(",", ":")),
        ),
        (
            TopologyQueryName.EVIDENCE_MENTIONS,
            json.dumps({"mentions": []}, separators=(",", ":")),
        ),
        (TopologyQueryName.RELATION_TOPOLOGY, "{"),
        (TopologyQueryName.EVIDENCE_MENTIONS, "{"),
    ),
)
def test_memgraph_loader_rejects_malformed_or_partial_family(query, section_json):
    driver = Driver(graph_snapshot())
    driver.section_overrides[query] = {"section_json": section_json}
    with pytest.raises(TopologyLoadError, match="direct_topology_invalid"):
        _load(driver)


def test_memgraph_loader_rejects_relation_endpoint_tampering() -> None:
    driver = Driver(graph_snapshot())
    relation = json.loads(
        driver._section(TopologyQueryName.RELATION_TOPOLOGY)["section_json"]
    )
    relation["relation_groups"][0]["source_identity_key"] = K[14]
    driver.section_overrides[TopologyQueryName.RELATION_TOPOLOGY] = {
        "section_json": json.dumps(relation, sort_keys=True, separators=(",", ":"))
    }
    with pytest.raises(TopologyLoadError, match="direct_topology_invalid"):
        _load(driver)


def test_memgraph_loader_rejects_family_caps_tampering() -> None:
    driver = Driver(graph_snapshot())
    membership = json.loads(
        driver._section(TopologyQueryName.AUTOMATIC_MEMBERSHIPS)["snapshot_json"]
    )
    membership["caps"]["max_nodes"] = 1
    driver.snapshot_json_override = json.dumps(
        membership, sort_keys=True, separators=(",", ":")
    )
    with pytest.raises(TopologyLoadError, match="direct_topology_invalid"):
        _load(driver)
