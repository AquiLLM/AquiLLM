"""Indexed topology reads preserve family-only corruption visibility."""

from __future__ import annotations

from dataclasses import replace
from time import monotonic

import pytest

from apps.knowledge_graph.projection.memgraph_driver import Neo4jMemgraphDriver
from apps.knowledge_graph.projection.memgraph_pagination import (
    initial_cursor_parameters,
)
from apps.knowledge_graph.projection.memgraph_records import FAMILIES
from apps.knowledge_graph.projection.memgraph_repository import (
    MemgraphProjectionRepository,
)
from apps.knowledge_graph.projection.topology_adapter import (
    Neo4jProjectedTopologyQueryAdapter,
)
from apps.knowledge_graph.projection.topology_cypher import bounded_family_query
from apps.knowledge_graph.retrieval.topology.contracts import (
    ProjectedSeedV1,
    TopologyFailureReason,
)
from apps.knowledge_graph.retrieval.topology.failures import TopologyLoadError
from apps.knowledge_graph.retrieval.topology.memgraph import (
    MemgraphProjectedTopologyLoader,
)
from apps.knowledge_graph.tests.memgraph_test_support import (
    isolated_memgraph_container as _isolated_memgraph_container,
)
from apps.knowledge_graph.tests.test_memgraph_projection_repository import (
    _expected_manifest,
)
from apps.knowledge_graph.tests.test_projected_topology_adapter import _caps, _ready
from apps.knowledge_graph.tests.test_projection_records import _bundle


@pytest.fixture
def isolated_memgraph_container():
    yield from _isolated_memgraph_container.__wrapped__()


@pytest.fixture
def staged_graph(isolated_memgraph_container):
    target = isolated_memgraph_container
    driver = Neo4jMemgraphDriver(target["uri"], "", "", database=target["database"])
    bundle = _bundle()
    try:
        repository = MemgraphProjectionRepository(driver)
        expected = replace(
            _expected_manifest(bundle), private_mapping_checksum="d" * 64
        )
        repository.write_staging_generation(
            bundle=bundle,
            private_mapping_checksum=expected.private_mapping_checksum,
            batch_size=128,
            timeout_seconds=5.0,
        )
        validation = repository.validate_generation(
            expected=expected, timeout_seconds=5.0
        )
        assert validation.valid
        repository.mark_generation_ready(
            generation_key=repository.opaque_generation_key(
                bundle.generation.generation_key
            ),
            validation_checksum=validation.validation_checksum,
            timeout_seconds=5.0,
        )
        # Existing node and edge indexes must remain safe on subsequent runs.
        repository.ensure_schema(timeout_seconds=5.0)
        yield driver, bundle
    finally:
        if driver._client is not None:
            driver._client.close()


def _parameters(bundle, label):
    return {
        "generation_key": bundle.generation.generation_key,
        "seed_keys_csv": bundle.entities[0].entity_key,
        "max_depth": 2,
        "authorized_document_keys_csv": ",".join(
            row.document_key for row in bundle.documents
        ),
        "collection_key": bundle.generation.collection_key,
        **initial_cursor_parameters(label),
        "page_limit": 100,
    }


def _load(driver, bundle):
    return MemgraphProjectedTopologyLoader(
        Neo4jProjectedTopologyQueryAdapter(driver)
    ).load(
        ready=_ready(bundle),
        seeds=(ProjectedSeedV1(bundle.entities[0].entity_key, 1.0),),
        caps=_caps(),
        deadline=monotonic() + 5.0,
    )


@pytest.mark.container
def test_all_topology_families_use_generation_indexes_and_preserve_results(
    staged_graph,
):
    driver, bundle = staged_graph
    expected_counts = {
        "ProjectedEntity": 2,
        "AutomaticMembership": 2,
        "ProjectedDocument": 1,
        "ProjectedChunk": 1,
        "ProjectedRelationSemantics": 1,
        "ProjectedRelation": 1,
        "ProjectedEvidence": 1,
        "ProjectedEntityMention": 1,
        "ArtifactProvenance": 2,
    }
    global_scans = []
    for label, _kind in FAMILIES:
        query = bounded_family_query(label)
        parameters = _parameters(bundle, label)
        plan = driver.execute_read(
            "EXPLAIN " + query, parameters, timeout_seconds=5.0, max_records=200
        )
        if any("ScanAll (" in row["QUERY PLAN"] for row in plan):
            global_scans.append(label)
        rows = driver.execute_read(
            query, parameters, timeout_seconds=5.0, max_records=100
        )
        assert len(rows) == expected_counts[label], label
    assert global_scans == []
    snapshot = _load(driver, bundle)
    assert len(snapshot.identity_keys) == 2
    assert len(snapshot.relation_groups) == 2  # Forward and reverse retrieval arcs.
    assert len(snapshot.mentions) == 1
    assert snapshot.allowed_scope.document_keys == (bundle.documents[0].document_key,)


@pytest.mark.container
def test_missing_shared_label_does_not_hide_malformed_in_scope_entity(staged_graph):
    driver, bundle = staged_graph
    assert _load(driver, bundle).relation_groups
    # This malformed family member is deliberately missing ProjectedRecord.
    # It shares a reachable identity, so the bounded rematch must still see it.
    driver.execute_write(
        "MATCH (n:ProjectedEntity {generation_key:$generation_key, "
        "opaque_key:$entity_key}) CREATE (bad:ProjectedEntity) "
        "SET bad=properties(n), bad.retrieval_utility=-1.0",
        {
            "generation_key": bundle.generation.generation_key,
            "entity_key": bundle.entities[0].entity_key,
        },
        timeout_seconds=5.0,
    )
    with pytest.raises(TopologyLoadError) as captured:
        _load(driver, bundle)
    assert captured.value.reason is TopologyFailureReason.BACKEND_SCHEMA_MISMATCH
