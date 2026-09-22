"""Chunk hydration follows the reached evidence neighborhood, not document size."""

from dataclasses import replace
from hashlib import sha256

import pytest

from apps.knowledge_graph.projection.memgraph_driver import Neo4jMemgraphDriver
from apps.knowledge_graph.projection.memgraph_pagination import (
    advance_cursor_parameters,
)
from apps.knowledge_graph.projection.memgraph_repository import (
    MemgraphProjectionRepository,
)
from apps.knowledge_graph.projection.topology_cypher import bounded_family_query
from apps.knowledge_graph.retrieval.topology.contracts import TopologyFailureReason
from apps.knowledge_graph.retrieval.topology.failures import TopologyLoadError
from apps.knowledge_graph.tests.memgraph_test_support import (
    isolated_memgraph_container as _isolated_memgraph_container,
)
from apps.knowledge_graph.tests.test_memgraph_projection_repository import (
    _expected_manifest,
)
from apps.knowledge_graph.tests.test_memgraph_topology_indexes import _load, _parameters
from apps.knowledge_graph.tests.test_projection_records import _bundle


@pytest.fixture
def isolated_memgraph_container():
    yield from _isolated_memgraph_container.__wrapped__()


@pytest.fixture
def large_document_graph(isolated_memgraph_container):
    target = isolated_memgraph_container
    driver = Neo4jMemgraphDriver(target["uri"], "", "", database=target["database"])
    original = _bundle()
    mention_chunk = replace(
        original.chunks[0],
        chunk_key=sha256(b"mention-chunk").hexdigest(),
        chunk_number=3,
    )
    unrelated = tuple(
        replace(
            original.chunks[0],
            chunk_key=sha256(f"unrelated:{index}".encode()).hexdigest(),
            chunk_number=index + 4,
        )
        for index in range(1500)
    )
    bundle = replace(
        original,
        chunks=(*original.chunks, mention_chunk, *unrelated),
        entity_mentions=(
            replace(
                original.entity_mentions[0],
                chunk_key=mention_chunk.chunk_key,
                chunk_number=mention_chunk.chunk_number,
            ),
        ),
        counts=replace(original.counts, chunk_count=1502),
    )
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
        yield driver, bundle
    finally:
        if driver._client is not None:
            driver._client.close()


def _chunk_rows(driver, bundle, *, page_limit=1000):
    return driver.execute_read(
        bounded_family_query("ProjectedChunk"),
        {**_parameters(bundle, "ProjectedChunk"), "page_limit": page_limit},
        timeout_seconds=5.0,
        max_records=page_limit,
    )


@pytest.mark.container
def test_large_document_hydrates_complete_referenced_chunks_with_keyset_pages(
    large_document_graph,
):
    driver, bundle = large_document_graph
    snapshot = _load(driver, bundle)
    assert len(snapshot.mentions) == 1
    assert len(snapshot.relation_groups) == 2
    expected = [bundle.evidence[0].chunk_key, bundle.entity_mentions[0].chunk_key]
    query = bounded_family_query("ProjectedChunk")
    parameters = {**_parameters(bundle, "ProjectedChunk"), "page_limit": 1}
    observed = []
    for _ in range(3):
        rows = driver.execute_read(
            query, parameters, timeout_seconds=5.0, max_records=1
        )
        if not rows:
            break
        properties = dict(rows[0]["record"])
        observed.append(properties["chunk_key"])
        parameters.update(
            advance_cursor_parameters(
                "ProjectedChunk", properties, rows[0]["cursor_id"]
            )
        )
    assert observed == expected
    assert rows == ()
    plan = driver.execute_read(
        "EXPLAIN " + query,
        _parameters(bundle, "ProjectedChunk"),
        timeout_seconds=5.0,
        max_records=200,
    )
    operators = "\n".join(row["QUERY PLAN"] for row in plan)
    assert "ScanAll (" not in operators
    assert ":ProjectedChunk {chunk_key}" in operators


@pytest.mark.container
def test_chunk_neighborhood_excludes_foreign_document_and_generation(
    large_document_graph,
):
    driver, bundle = large_document_graph
    driver.execute_write(
        "MATCH (entity:ProjectedEntity {generation_key:$generation_key, "
        "entity_key:$entity_key}), "
        "(document:ProjectedDocument {generation_key:$generation_key, "
        "document_key:$document_key}) "
        "CREATE (foreign:ProjectedDocument {generation_key:$generation_key, "
        "document_key:$foreign_key}) "
        "WITH entity, document, foreign "
        "UNWIND range(0,2) AS i "
        "CREATE (chunk:ProjectedChunk {generation_key:CASE WHEN i=1 "
        "THEN $foreign_key ELSE $generation_key END, "
        "document_key:$document_key, chunk_key:toString(i), opaque_key:toString(i), "
        "chunk_number:i}) "
        "CREATE (entity)-[:ENTITY_MENTION {generation_key:CASE WHEN i=2 "
        "THEN $foreign_key ELSE $generation_key END}]->(chunk) "
        "WITH document, foreign, chunk, i "
        "FOREACH (ignored IN CASE WHEN i=0 THEN [1] ELSE [] END | "
        "CREATE (foreign)-[:DOCUMENT_CHUNK {generation_key:$generation_key}]->(chunk)) "
        "FOREACH (ignored IN CASE WHEN i<>0 THEN [1] ELSE [] END | "
        "CREATE (document)-[:DOCUMENT_CHUNK "
        "{generation_key:$generation_key}]->(chunk))",
        {
            "generation_key": bundle.generation.generation_key,
            "entity_key": bundle.entities[0].entity_key,
            "document_key": bundle.documents[0].document_key,
            "foreign_key": "f" * 64,
        },
        timeout_seconds=5.0,
    )
    assert [
        dict(row["record"])["chunk_key"] for row in _chunk_rows(driver, bundle)
    ] == [
        bundle.evidence[0].chunk_key,
        bundle.entity_mentions[0].chunk_key,
    ]


@pytest.mark.container
def test_chunk_neighborhood_preserves_hop_bound(large_document_graph):
    driver, bundle = large_document_graph
    far_chunk = bundle.chunks[2]
    driver.execute_write(
        "MATCH (entity:ProjectedEntity {generation_key:$generation_key, "
        "entity_key:$entity_key}), "
        "(chunk:ProjectedChunk {generation_key:$generation_key, chunk_key:$chunk_key}) "
        "CREATE (far:ProjectedEntity {generation_key:$generation_key}) "
        "CREATE (entity)-[:PROJECTED_RELATION {generation_key:$generation_key}]->(far) "
        "CREATE (far)-[:ENTITY_MENTION {generation_key:$generation_key}]->(chunk)",
        {
            "generation_key": bundle.generation.generation_key,
            "entity_key": bundle.entities[1].entity_key,
            "chunk_key": far_chunk.chunk_key,
        },
        timeout_seconds=5.0,
    )
    for depth, expected in ((1, 2), (2, 3)):
        rows = driver.execute_read(
            bounded_family_query("ProjectedChunk"),
            {**_parameters(bundle, "ProjectedChunk"), "max_depth": depth},
            timeout_seconds=5.0,
            max_records=100,
        )
        assert len(rows) == expected
        assert (
            far_chunk.chunk_key in {dict(row["record"])["chunk_key"] for row in rows}
        ) is (depth == 2)


@pytest.mark.container
def test_referenced_duplicate_chunk_without_shared_label_still_fails_closed(
    large_document_graph,
):
    driver, bundle = large_document_graph
    driver.execute_write(
        "MATCH (document:ProjectedDocument {generation_key:$generation_key})"
        "-[:DOCUMENT_CHUNK {generation_key:$generation_key}]->"
        "(chunk:ProjectedChunk {generation_key:$generation_key, chunk_key:$chunk_key}) "
        "CREATE (bad:ProjectedChunk) SET bad=properties(chunk) "
        "CREATE (document)-[:DOCUMENT_CHUNK {generation_key:$generation_key}]->(bad)",
        {
            "generation_key": bundle.generation.generation_key,
            "chunk_key": bundle.evidence[0].chunk_key,
        },
        timeout_seconds=5.0,
    )
    with pytest.raises(TopologyLoadError) as captured:
        _load(driver, bundle)
    assert captured.value.reason is TopologyFailureReason.BACKEND_SCHEMA_MISMATCH
