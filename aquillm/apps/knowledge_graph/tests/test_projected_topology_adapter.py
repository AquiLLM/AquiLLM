from __future__ import annotations

from dataclasses import asdict, replace
from hashlib import sha256

import pytest

from apps.knowledge_graph.projection.memgraph_driver import MemgraphDriverError
from apps.knowledge_graph.projection.serialization import projection_checksum
from apps.knowledge_graph.projection.topology_adapter import (
    Neo4jProjectedTopologyQueryAdapter,
)
from apps.knowledge_graph.retrieval.ppr import (
    RetrievalDirection,
    raw_edge_weight,
)
from apps.knowledge_graph.retrieval.projected_types import (
    ProjectedRetrievalDirectionV1,
)
from apps.knowledge_graph.retrieval.topology.contracts import (
    AuthorizedProjectedDocumentV1,
    HybridBranchKind,
    ProjectedSeedV1,
    ReadyGenerationBundleV1,
    SelectedCollectionGenerationV1,
    TopologyCapsV1,
    TopologyFailureReason,
    TopologyQueryName,
    ready_generation_bundle_checksum,
)
from apps.knowledge_graph.retrieval.topology.failures import TopologyLoadError
from apps.knowledge_graph.retrieval.topology.memgraph import (
    MemgraphProjectedTopologyLoader,
    _parameters,
)
from apps.knowledge_graph.tests.test_projection_records import _bundle

_FAMILIES = (
    ("ProjectedEntity", "entities", "entity_key"),
    ("AutomaticMembership", "automatic_memberships", "entity_key"),
    ("ProjectedDocument", "documents", "document_key"),
    ("ProjectedChunk", "chunks", "chunk_key"),
    ("ProjectedRelationSemantics", "relation_semantics", "semantics_key"),
    ("ProjectedRelation", "relations", "relation_key"),
    ("ProjectedEvidence", "evidence", "evidence_key"),
    ("ProjectedEntityMention", "entity_mentions", "mention_key"),
    ("ArtifactProvenance", "artifact_provenance", "scope_key"),
)


def _ready(bundle):
    collection = next(
        row for row in bundle.artifact_provenance if row.scope_type == "collection"
    )
    marker = bundle.generation
    selected = SelectedCollectionGenerationV1(
        marker.collection_key,
        marker.generation_key,
        marker.artifact_key,
        marker.projection_key,
        projection_checksum(bundle),
        marker.schema_version,
        marker.projection_version,
        marker.identifier_key_version,
        marker.membership_epoch,
        marker.membership_checksum,
        collection.resolver_version,
        collection.resolution_config_checksum,
        collection.ontology_checksum,
        collection.embedding_model_signature,
    )
    documents = tuple(
        AuthorizedProjectedDocumentV1(
            row.document_key, marker.collection_key, marker.generation_key
        )
        for row in bundle.documents
    )
    signature = sha256(b"authorization").hexdigest()
    return ReadyGenerationBundleV1(
        (selected,),
        documents,
        signature,
        ready_generation_bundle_checksum((selected,), documents, signature),
    )


class ProjectionDriver:
    def __init__(self, bundle=None, *, error_code=None, manifest_checksum=None):
        self.bundle = _bundle() if bundle is None else bundle
        self.error_code = error_code
        self.manifest_checksum = manifest_checksum
        self.calls = []

    def execute_read(self, cypher, parameters, *, timeout_seconds, max_records):
        self.calls.append((cypher, parameters, timeout_seconds, max_records))
        if self.error_code is not None:
            raise MemgraphDriverError(self.error_code)
        if "AS collection_key" in cypher:
            marker = self.bundle.generation
            return (
                {
                    "collection_key": marker.collection_key,
                    "generation_key": marker.generation_key,
                    "projection_key": marker.projection_key,
                    "active_artifact_key": marker.artifact_key,
                    "graph_checksum": self.manifest_checksum
                    or projection_checksum(self.bundle),
                    "membership_checksum": marker.membership_checksum,
                    "state": "ready",
                },
            )
        if "CollectionGeneration" in cypher:
            return ({"record": asdict(self.bundle.generation)},)
        for label, field, identity in _FAMILIES:
            if f"n:{label} " in cypher:
                return tuple(
                    {
                        "record": {
                            **asdict(row),
                            "generation_key": self.bundle.generation.generation_key,
                            "opaque_key": getattr(row, identity),
                        }
                    }
                    for row in getattr(self.bundle, field)
                )
        raise AssertionError("adapter emitted an unknown Cypher query")


def _caps():
    return TopologyCapsV1(HybridBranchKind.DIRECT, 32, 2, 200, 1_000, 20)


def test_adapter_builds_actual_authorized_bfs_families_with_fixed_cypher() -> None:
    bundle = _bundle()
    ready = _ready(bundle)
    seed_identity = bundle.entities[0].entity_key
    driver = ProjectionDriver(bundle)
    adapter = Neo4jProjectedTopologyQueryAdapter(driver, clock=lambda: 40.0)

    snapshot = MemgraphProjectedTopologyLoader(adapter).load(
        ready=ready,
        seeds=(ProjectedSeedV1(seed_identity, 1.0),),
        caps=_caps(),
        deadline=42.5,
    )

    group = snapshot.relation_groups[0]
    assert (group.source_identity_key, group.target_identity_key) == (
        seed_identity,
        bundle.automatic_memberships[1].automatic_membership_key,
    )
    assert group.direction is ProjectedRetrievalDirectionV1.FORWARD
    assert group.raw_weight == raw_edge_weight(
        direction=RetrievalDirection.FORWARD,
        confidence=bundle.evidence[0].confidence,
        support_count=1,
        destination_retrieval_utility=bundle.entities[1].retrieval_utility,
    )
    assert snapshot.mentions[0].identity_key == seed_identity
    assert snapshot.allowed_scope.document_keys == (bundle.documents[0].document_key,)
    assert snapshot.load_max_hops == 2
    assert all(seed_identity not in cypher for cypher, *_rest in driver.calls)
    assert {tuple(call[1]) for call in driver.calls} == {("generation_key",)}
    assert {call[2] for call in driver.calls} == {2.5}


def test_adapter_maps_fixed_backend_failures_and_expired_deadline() -> None:
    bundle = _bundle()
    ready = _ready(bundle)
    seeds = (ProjectedSeedV1(bundle.entities[0].entity_key, 1.0),)
    parameters = _parameters(ready, seeds, _caps())
    expired = ProjectionDriver(bundle)
    adapter = Neo4jProjectedTopologyQueryAdapter(expired, clock=lambda: 43.0)

    with pytest.raises(TimeoutError):
        adapter.execute_read(
            query=TopologyQueryName.GENERATION_MANIFESTS,
            parameters=parameters,
            deadline=42.5,
            max_records=1,
        )
    assert expired.calls == []

    auth = Neo4jProjectedTopologyQueryAdapter(
        ProjectionDriver(bundle, error_code="memgraph_authentication_failed"),
        clock=lambda: 40.0,
    )
    with pytest.raises(TopologyLoadError) as captured:
        auth.execute_read(
            query=TopologyQueryName.GENERATION_MANIFESTS,
            parameters=parameters,
            deadline=42.5,
            max_records=1,
        )
    assert captured.value.reason is TopologyFailureReason.BACKEND_AUTHENTICATION


def test_adapter_rejects_checksum_drift_and_arbitrary_query_passthrough() -> None:
    original = _bundle()
    changed = replace(
        original,
        entities=(
            replace(original.entities[0], retrieval_utility=0.625),
            original.entities[1],
        ),
    )
    driver = ProjectionDriver(changed, manifest_checksum=projection_checksum(original))
    ready = _ready(original)
    adapter = Neo4jProjectedTopologyQueryAdapter(driver, clock=lambda: 40.0)

    with pytest.raises(TopologyLoadError) as captured:
        MemgraphProjectedTopologyLoader(adapter).load(
            ready=ready,
            seeds=(ProjectedSeedV1(original.entities[0].entity_key, 1.0),),
            caps=_caps(),
            deadline=42.5,
        )
    assert captured.value.reason is TopologyFailureReason.BACKEND_PROVENANCE_MISMATCH

    calls = len(driver.calls)
    with pytest.raises(TypeError):
        adapter.execute_read(
            query="MATCH (n) RETURN n",  # type: ignore[arg-type]
            parameters={},
            deadline=42.5,
            max_records=1,
        )
    assert len(driver.calls) == calls


def test_adapter_uses_projected_undirected_semantics_in_both_directions() -> None:
    original = _bundle()
    bundle = replace(
        original,
        relation_semantics=(
            replace(original.relation_semantics[0], direction="undirected"),
        ),
    )
    ready = _ready(bundle)

    snapshot = MemgraphProjectedTopologyLoader(
        Neo4jProjectedTopologyQueryAdapter(ProjectionDriver(bundle), clock=lambda: 40.0)
    ).load(
        ready=ready,
        seeds=(ProjectedSeedV1(bundle.entities[0].entity_key, 1.0),),
        caps=_caps(),
        deadline=42.5,
    )

    assert {row.direction for row in snapshot.relation_groups} == {
        ProjectedRetrievalDirectionV1.UNDIRECTED
    }


def test_adapter_rejects_selected_metadata_not_attested_by_projection() -> None:
    bundle = _bundle()
    actual = _ready(bundle)
    selected = replace(actual.selected_generations[0], resolver_version="resolver-v2")
    ready = ReadyGenerationBundleV1(
        (selected,),
        actual.authorized_documents,
        actual.authorization_context_signature,
        ready_generation_bundle_checksum(
            (selected,),
            actual.authorized_documents,
            actual.authorization_context_signature,
        ),
    )

    with pytest.raises(TopologyLoadError) as captured:
        MemgraphProjectedTopologyLoader(
            Neo4jProjectedTopologyQueryAdapter(
                ProjectionDriver(bundle), clock=lambda: 40.0
            )
        ).load(
            ready=ready,
            seeds=(ProjectedSeedV1(bundle.entities[0].entity_key, 1.0),),
            caps=_caps(),
            deadline=42.5,
        )

    assert captured.value.reason is TopologyFailureReason.BACKEND_PROVENANCE_MISMATCH
