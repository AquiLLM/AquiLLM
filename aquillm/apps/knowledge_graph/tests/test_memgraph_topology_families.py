from __future__ import annotations

import json
from dataclasses import replace
from hashlib import sha256

import pytest

from apps.knowledge_graph.retrieval import projected_types as t
from apps.knowledge_graph.retrieval.branch_contracts import (
    BranchStatusV1,
    DirectBranchFailureReason,
    ExtendedBranchFailureReason,
)
from apps.knowledge_graph.retrieval.scheduler import HybridGraphBranchScheduler
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
from apps.knowledge_graph.tests.test_retrieval_branch_scheduler import (
    _Runtime,
    _settings,
)


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


def _load(driver, branch=HybridBranchKind.DIRECT):
    return MemgraphProjectedTopologyLoader(driver).load(
        ready=ready(),
        seeds=(ProjectedSeedV1(K[4], 1.0),),
        caps=TopologyCapsV1(branch, 32, 2, 200, 1_000, 20),
        deadline=42.5,
    )


def _overflow_driver(query, path):
    driver = Driver(graph_snapshot())
    section = json.loads(driver._section(query)["section_json"])
    target = section
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = "0x1p+999999999"
    driver.section_overrides[query] = {
        "section_json": json.dumps(
            section,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    }
    return driver


def test_memgraph_loader_constructs_graph_from_all_response_families() -> None:
    graph = graph_snapshot()
    assert _load(Driver(graph)) == graph


def test_memgraph_loader_accepts_canonical_literal_utf8_families() -> None:
    graph = graph_snapshot()
    provenance = (
        replace(graph.artifact_provenance[0], ontology_version="ontología-v1"),
        *graph.artifact_provenance[1:],
    )
    graph = replace(graph, artifact_provenance=provenance)

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


@pytest.mark.parametrize("branch", tuple(HybridBranchKind))
@pytest.mark.parametrize(
    ("query", "path"),
    (
        (TopologyQueryName.RELATION_TOPOLOGY, ("relation_groups", 0, "raw_weight")),
        (
            TopologyQueryName.RELATION_TOPOLOGY,
            ("relation_groups", 0, "evidence", 0, "confidence"),
        ),
        (
            TopologyQueryName.EVIDENCE_MENTIONS,
            ("audit_rows", 0, "signature", "confidence"),
        ),
    ),
)
def test_memgraph_loader_normalizes_overflowed_hex_floats_to_local_invalid(
    branch, query, path
) -> None:
    reason = (
        "direct_topology_invalid"
        if branch is HybridBranchKind.DIRECT
        else "extended_topology_invalid"
    )

    with pytest.raises(TopologyLoadError, match=reason):
        _load(_overflow_driver(query, path), branch)


@pytest.mark.parametrize("failing_branch", tuple(HybridBranchKind))
def test_scheduler_preserves_sibling_for_overflowed_backend_json(
    failing_branch,
) -> None:
    runtime = _Runtime()

    def overflow(**_kwargs):
        return _load(
            _overflow_driver(
                TopologyQueryName.RELATION_TOPOLOGY,
                ("relation_groups", 0, "raw_weight"),
            ),
            failing_branch,
        )

    if failing_branch is HybridBranchKind.DIRECT:
        runtime.run_direct = overflow
        sibling_name, failed_name = "extended", "direct"
        expected = DirectBranchFailureReason.DIRECT_TOPOLOGY_INVALID
    else:
        runtime.run_extended = overflow
        sibling_name, failed_name = "direct", "extended"
        expected = ExtendedBranchFailureReason.EXTENDED_TOPOLOGY_INVALID
    outcome = HybridGraphBranchScheduler(runtime, clock=lambda: 40.0).run(
        query="q",
        baseline=object(),
        authorization=object(),
        settings=_settings(),
        deadline=42.5,
    )

    assert getattr(outcome, sibling_name).status is BranchStatusV1.SUCCEEDED
    assert getattr(outcome, failed_name).failure_reason is expected
    assert outcome.shared_failure_reason is None
