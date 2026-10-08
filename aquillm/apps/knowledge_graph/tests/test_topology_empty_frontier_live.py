"""Write fixtures only to an explicitly opted-in disposable Memgraph container."""

import os
from dataclasses import replace
from hashlib import sha256
from time import monotonic

import pytest

from apps.knowledge_graph.projection.memgraph_driver import Neo4jMemgraphDriver
from apps.knowledge_graph.projection.memgraph_repository import (
    MemgraphProjectionRepository,
)
from apps.knowledge_graph.projection.topology_adapter import (
    Neo4jProjectedTopologyQueryAdapter,
)
from apps.knowledge_graph.retrieval.topology.contracts import (
    ProjectedSeedV1,
    ReadyGenerationBundleV1,
    ready_generation_bundle_checksum,
)
from apps.knowledge_graph.retrieval.topology.failures import TopologyLoadError
from apps.knowledge_graph.retrieval.topology.memgraph import (
    MemgraphProjectedTopologyLoader,
)
from apps.knowledge_graph.tests.test_memgraph_projection_repository import (
    _expected_manifest,
)
from apps.knowledge_graph.tests.test_projected_topology_adapter import _caps, _ready
from apps.knowledge_graph.tests.test_projection_records import _bundle
from apps.knowledge_graph.tests.test_topology_empty_frontier import empty_bundle

pytestmark = pytest.mark.skipif(
    os.environ.get("KG_DISPOSABLE_FRONTIER_TESTS") != "1",
    reason="requires isolated kg-reliability-memgraph container",
)


@pytest.fixture
def graph():
    # Fixed internal endpoint cannot accidentally target configured live graphs.
    driver = Neo4jMemgraphDriver(
        "bolt://kg-reliability-memgraph:7687",
        "",
        "",
        database="memgraph",
        max_transaction_retry_time=0.0,
    )
    repository = MemgraphProjectionRepository(driver)
    bundle = _bundle()
    key = repository.opaque_generation_key(bundle.generation.generation_key)
    repository.ensure_schema(timeout_seconds=5.0)
    repository.delete_generation(generation_key=key, timeout_seconds=5.0)
    try:
        # A projection with chunks requires a nonempty private-map checksum.
        expected = replace(
            _expected_manifest(bundle),
            private_mapping_checksum=sha256(b"synthetic-private-map").hexdigest(),
        )
        repository.write_staging_generation(
            bundle=bundle,
            private_mapping_checksum=expected.private_mapping_checksum,
            batch_size=2,
            timeout_seconds=5.0,
        )
        validation = repository.validate_generation(
            expected=expected, timeout_seconds=5.0
        )
        assert validation.valid
        repository.mark_generation_ready(
            generation_key=key,
            validation_checksum=validation.validation_checksum,
            timeout_seconds=5.0,
        )
        yield driver, bundle
    finally:
        repository.delete_generation(generation_key=key, timeout_seconds=5.0)
        driver._connection().close()


class RecordingDriver:
    def __init__(self, driver, *, force_nonempty=False):
        self.driver, self.force_nonempty, self.calls = driver, force_nonempty, []

    def execute_read(self, query, parameters, **kwargs):
        rows = self.driver.execute_read(query, parameters, **kwargs)
        self.calls.append((query, rows))
        if self.force_nonempty and "AS frontier_count" in query:
            return ({"frontier_count": 1},)
        return rows


def load(driver, bundle, seed, ready=None):
    return MemgraphProjectedTopologyLoader(
        Neo4jProjectedTopologyQueryAdapter(driver)
    ).load(
        ready=_ready(bundle) if ready is None else ready,
        seeds=(ProjectedSeedV1(seed, 1.0),),
        caps=_caps(),
        deadline=monotonic() + 5.0,
    )


@pytest.mark.parametrize("mutation", ["REMOVE n.opaque_key", "SET n.opaque_key = null"])
def test_physical_malformed_entities_cannot_be_mistaken_for_empty_frontier(
    graph, mutation
):
    driver, bundle = graph
    driver.execute_write(
        "MATCH (n:ProjectedEntity {generation_key:$generation_key}) " + mutation,
        {"generation_key": bundle.generation.generation_key},
        timeout_seconds=5.0,
    )
    recording = RecordingDriver(driver)
    with pytest.raises(TopologyLoadError):
        load(recording, bundle, bundle.entities[0].entity_key)
    assert next(
        rows for query, rows in recording.calls if "AS frontier_count" in query
    ) == ({"frontier_count": 1},)
    # Linked valid membership, mention and physical-relation records remain read.
    for label in ("AutomaticMembership", "ProjectedEntityMention", "ProjectedRelation"):
        assert any(f"(n:{label} " in query and rows for query, rows in recording.calls)


def test_disconnected_duplicate_identity_still_rejected(graph):
    driver, bundle = graph
    driver.execute_write(
        "MATCH (n:ProjectedEntity {generation_key:$generation_key}) "
        "WITH n LIMIT 1 CREATE (duplicate:ProjectedEntity) "
        "SET duplicate = properties(n)",
        {"generation_key": bundle.generation.generation_key},
        timeout_seconds=5.0,
    )
    with pytest.raises(TopologyLoadError):
        load(RecordingDriver(driver), bundle, bundle.entities[0].entity_key)


def test_empty_frontier_canonical_equivalence_and_authorized_scope(graph):
    from apps.knowledge_graph.retrieval.projected_types import (
        canonical_projected_snapshot_bytes,
    )

    driver, bundle = graph
    guarded, original = (
        RecordingDriver(driver),
        RecordingDriver(driver, force_nonempty=True),
    )

    # An empty generation must coexist with a nonempty sibling: a completely
    # empty final snapshot is intentionally invalid under the existing contract.
    def key(name):
        return sha256(name.encode()).hexdigest()

    sibling = empty_bundle()
    sibling = replace(
        sibling,
        generation=replace(
            sibling.generation,
            generation_key=key("empty-generation"),
            artifact_key=key("empty-collection-artifact"),
            collection_key=key("empty-collection"),
            projection_key=key("empty-projection"),
        ),
        documents=(
            replace(
                sibling.documents[0],
                generation_key=key("empty-generation"),
                document_key=key("empty-document"),
            ),
        ),
        artifact_provenance=tuple(
            sorted(
                (
                    replace(
                        row,
                        collection_key=key("empty-collection"),
                        artifact_key=key("empty-" + row.scope_type + "-artifact"),
                        scope_key=key("empty-collection")
                        if row.scope_type == "collection"
                        else key("empty-document"),
                    )
                    for row in sibling.artifact_provenance
                ),
                key=lambda row: (row.scope_type, row.scope_key, row.artifact_key),
            )
        ),
    )
    repository = MemgraphProjectionRepository(driver)
    sibling_key = repository.opaque_generation_key(sibling.generation.generation_key)
    repository.delete_generation(generation_key=sibling_key, timeout_seconds=5.0)
    try:
        expected = _expected_manifest(sibling)
        repository.write_staging_generation(
            bundle=sibling,
            private_mapping_checksum=expected.private_mapping_checksum,
            batch_size=2,
            timeout_seconds=5.0,
        )
        validation = repository.validate_generation(
            expected=expected, timeout_seconds=5.0
        )
        assert validation.valid
        repository.mark_generation_ready(
            generation_key=sibling_key,
            validation_checksum=validation.validation_checksum,
            timeout_seconds=5.0,
        )
        first, second = _ready(bundle), _ready(sibling)
        generations = tuple(
            sorted(
                (*first.selected_generations, *second.selected_generations),
                key=lambda row: row.collection_key,
            )
        )
        documents = tuple(
            sorted(
                (*first.authorized_documents, *second.authorized_documents),
                key=lambda row: row.document_key,
            )
        )
        ready = ReadyGenerationBundleV1(
            generations,
            documents,
            first.authorization_context_signature,
            ready_generation_bundle_checksum(
                generations, documents, first.authorization_context_signature
            ),
        )
        empty = load(guarded, bundle, bundle.entities[0].entity_key, ready)
        reference = load(original, bundle, bundle.entities[0].entity_key, ready)
    finally:
        repository.delete_generation(generation_key=sibling_key, timeout_seconds=5.0)
    assert canonical_projected_snapshot_bytes(
        empty
    ) == canonical_projected_snapshot_bytes(reference)
    assert empty.allowed_scope.document_keys == tuple(
        sorted((bundle.documents[0].document_key, sibling.documents[0].document_key))
    )
    assert empty.artifact_provenance
    assert len(guarded.calls) == 16  # manifest + nonempty10 + guarded empty5
    assert len(original.calls) == 22
