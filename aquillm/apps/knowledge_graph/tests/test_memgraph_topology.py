from __future__ import annotations

import json
from dataclasses import replace

import pytest

from apps.knowledge_graph.retrieval import projected_types as t
from apps.knowledge_graph.retrieval.topology.contracts import (
    AuthorizedProjectedDocumentV1,
    HybridBranchKind,
    ProjectedSeedV1,
    ReadyGenerationBundleV1,
    SelectedCollectionGenerationV1,
    TopologyCapsV1,
    TopologyQueryName,
    ready_generation_bundle_checksum,
)
from apps.knowledge_graph.retrieval.topology.memgraph import (
    MemgraphProjectedTopologyLoader,
    TopologyLoadError,
)

K = tuple(character * 64 for character in "123456789abcdef")


def _provenance(scope: t.ProjectedScopeTypeV1):
    collection = K[0]
    is_collection = scope is t.ProjectedScopeTypeV1.COLLECTION
    return t.ProjectedArtifactProvenanceV1(
        K[5] if is_collection else K[6],
        scope,
        collection if is_collection else K[2],
        collection,
        None,
        False,
        K[7] if is_collection else K[8],
        1,
        1,
        K[9] if is_collection else K[10],
        "ontology-v1",
        K[11],
        "extractor-v1",
        "resolver-v1",
        K[12],
        "filter-v1",
        K[13],
        "embed-v1" if is_collection else "",
        "assembly-v1",
        K[14],
    )


def snapshot():
    membership = t.ProjectedAutomaticMembershipAuditV1(
        0, K[3], K[4], K[9], "resolver-v1"
    )
    return t.ProjectedAuthorizedGraphSnapshotV1(
        t.ProjectedAlgorithmSignatureV1(
            "ppr_projected_v1",
            "ppr_transition_v1",
            "ppr_evidence_v1",
            "rrf_seed_v1",
            K[1],
        ),
        t.ProjectedSnapshotCapsV1(64, 10_000, 128, 2, 200, 1_000, 3_000, 3, 2),
        2,
        t.ProjectedAllowedScopeV1((K[2],), (K[0],), K[1]),
        (K[4],),
        (),
        (),
        (),
        (
            _provenance(t.ProjectedScopeTypeV1.COLLECTION),
            _provenance(t.ProjectedScopeTypeV1.DOCUMENT),
        ),
        (membership,),
    )


def ready():
    generation = SelectedCollectionGenerationV1(
        K[0],
        K[1],
        K[5],
        K[6],
        K[7],
        "schema-v1",
        "projection-v1",
        "key-v1",
        3,
        K[8],
        "resolver-v1",
        K[12],
        K[11],
        "embed-v1",
    )
    documents = (AuthorizedProjectedDocumentV1(K[2], K[0], K[1]),)
    signature = K[13]
    return ReadyGenerationBundleV1(
        (generation,),
        documents,
        signature,
        ready_generation_bundle_checksum((generation,), documents, signature),
    )


class Driver:
    def __init__(self, graph=None):
        self.graph = graph or snapshot()
        self.calls = []
        self.manifest_override = None
        self.snapshot_json_override = None
        self.section_overrides = {}

    def _section(self, query):
        value = json.loads(t.canonical_projected_snapshot_bytes(self.graph))
        audits = value["audit_rows"]
        if query is TopologyQueryName.AUTOMATIC_MEMBERSHIPS:
            value["relation_groups"] = []
            value["mentions"] = []
            value["audit_rows"] = [
                row for row in audits if row["kind"] == "automatic_membership"
            ]
            return {
                "snapshot_json": json.dumps(
                    value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
            }
        if query is TopologyQueryName.RELATION_TOPOLOGY:
            section = {
                "relation_groups": value["relation_groups"],
                "audit_rows": [
                    row for row in audits if row["kind"] == "physical_relation"
                ],
            }
        else:
            section = {
                "mentions": value["mentions"],
                "audit_rows": [
                    row
                    for row in audits
                    if row["kind"] in {"fallback_mention", "relation_evidence"}
                ],
            }
        return {
            "section_json": json.dumps(
                section, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
        }

    def execute_read(self, *, query, parameters, deadline, max_records):
        self.calls.append((query, parameters, deadline, max_records))
        if query is TopologyQueryName.GENERATION_MANIFESTS:
            if self.manifest_override is not None:
                return self.manifest_override
            row = ready().selected_generations[0]
            return (
                {
                    "collection_key": row.collection_key,
                    "generation_key": row.generation_key,
                    "projection_key": row.projection_key,
                    "active_artifact_key": row.active_artifact_key,
                    "graph_checksum": row.graph_checksum,
                    "membership_checksum": row.membership_checksum,
                },
            )
        if query is TopologyQueryName.AUTOMATIC_MEMBERSHIPS:
            row = self._section(query)
            if self.snapshot_json_override is not None:
                row["snapshot_json"] = self.snapshot_json_override
            return (row,)
        return (self.section_overrides.get(query, self._section(query)),)


def test_memgraph_loader_uses_exact_parameterized_scope_deadline_and_caps() -> None:
    driver = Driver()
    loader = MemgraphProjectedTopologyLoader(driver)
    seeds = (ProjectedSeedV1(K[4], 1.0),)
    caps = TopologyCapsV1(HybridBranchKind.DIRECT, 32, 2, 200, 1_000, 20)
    result = loader.load(ready=ready(), seeds=seeds, caps=caps, deadline=42.5)
    assert result == snapshot()
    assert tuple(call[0] for call in driver.calls) == tuple(TopologyQueryName)
    assert {call[2] for call in driver.calls} == {42.5}
    assert tuple(call[3] for call in driver.calls) == (1, 200, 1_000, 3_400)
    for _, parameters, _, _ in driver.calls:
        assert parameters["bundle_checksum"] == ready().bundle_checksum
        assert parameters["generation_keys_json"] == f'["{K[1]}"]'
        assert parameters["document_keys_json"] == f'["{K[2]}"]'
        assert parameters["membership_checksums_json"] == f'["{K[8]}"]'


@pytest.mark.parametrize("field", ("generation_key", "membership_checksum"))
def test_memgraph_loader_is_all_or_nothing_for_ready_manifest(field: str) -> None:
    driver = Driver()
    row = dict(
        Driver().execute_read(
            query=TopologyQueryName.GENERATION_MANIFESTS,
            parameters={},
            deadline=1.0,
            max_records=1,
        )[0]
    )
    row[field] = K[14]
    driver.manifest_override = (row,)
    with pytest.raises(TopologyLoadError, match="readiness_mismatch"):
        MemgraphProjectedTopologyLoader(driver).load(
            ready=ready(),
            seeds=(ProjectedSeedV1(K[4], 1.0),),
            caps=TopologyCapsV1(HybridBranchKind.DIRECT, 32, 2, 200, 1_000, 20),
            deadline=42.5,
        )


def test_memgraph_loader_rejects_snapshot_scope_or_endpoint_tampering() -> None:
    driver = Driver()
    driver.snapshot_json_override = (
        t.canonical_projected_snapshot_bytes(snapshot()).decode().replace(K[2], K[3])
    )
    with pytest.raises(TopologyLoadError, match="direct_topology_invalid"):
        MemgraphProjectedTopologyLoader(driver).load(
            ready=ready(),
            seeds=(ProjectedSeedV1(K[4], 1.0),),
            caps=TopologyCapsV1(HybridBranchKind.DIRECT, 32, 2, 200, 1_000, 20),
            deadline=42.5,
        )


@pytest.mark.parametrize(
    ("loaded_depth", "requested_depth", "rejected"),
    ((2, 1, True), (1, 2, False)),
)
def test_memgraph_loader_enforces_requested_depth_cap(
    loaded_depth: int, requested_depth: int, rejected: bool
) -> None:
    driver = Driver(replace(snapshot(), load_max_hops=loaded_depth))

    def load():
        return MemgraphProjectedTopologyLoader(driver).load(
            ready=ready(),
            seeds=(ProjectedSeedV1(K[4], 1.0),),
            caps=TopologyCapsV1(
                HybridBranchKind.DIRECT, 32, requested_depth, 200, 1_000, 20
            ),
            deadline=42.5,
        )

    if rejected:
        with pytest.raises(TopologyLoadError, match="direct_topology_invalid"):
            load()
    else:
        assert load().load_max_hops == loaded_depth
