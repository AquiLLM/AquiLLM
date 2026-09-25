from __future__ import annotations

import pytest

from apps.knowledge_graph.projection import memgraph_edge_validation as edges
from apps.knowledge_graph.projection.memgraph_driver import Neo4jMemgraphDriver
from apps.knowledge_graph.projection.memgraph_edges import (
    topology_edge_attestation,
    write_topology_edges,
)
from apps.knowledge_graph.projection.memgraph_repository import (
    MemgraphProjectionRepository,
)
from apps.knowledge_graph.tests.memgraph_test_support import (
    isolated_memgraph_container as _isolated_memgraph_container,
)
from apps.knowledge_graph.tests.test_projection_records import _bundle


@pytest.fixture
def isolated_memgraph_container():
    yield from _isolated_memgraph_container.__wrapped__()


_SPECS = (
    (
        "ENTITY_MEMBERSHIP",
        "coalesce(source.opaque_key, '')",
        "source.opaque_key AS entity_key, "
        "target.automatic_membership_key AS automatic_membership_key",
    ),
    (
        "DOCUMENT_CHUNK",
        "coalesce(target.opaque_key, '')",
        "source.opaque_key AS document_key, target.opaque_key AS chunk_key, "
        "edge.chunk_number AS chunk_number",
    ),
    (
        "PROJECTED_RELATION",
        "coalesce(edge.relation_key, '')",
        "edge.relation_key AS relation_key, edge.artifact_key AS artifact_key, "
        "source.opaque_key AS source_entity_key, edge.relation_type AS relation_type, "
        "target.opaque_key AS target_entity_key, edge.direction AS direction",
    ),
    (
        "RELATION_EVIDENCE",
        "coalesce(edge.evidence_key, '')",
        "edge.evidence_key AS evidence_key, source.opaque_key AS relation_key, "
        "edge.relation_mention_key AS relation_mention_key, "
        "target.opaque_key AS chunk_key, edge.document_key AS document_key, "
        "edge.chunk_number AS chunk_number, edge.confidence AS confidence, "
        "edge.provenance_key AS provenance_key, "
        "edge.semantic_signature AS semantic_signature, "
        "edge.head_mention_key AS head_mention_key, "
        "edge.tail_mention_key AS tail_mention_key, edge.orientation AS orientation",
    ),
    (
        "ENTITY_MENTION",
        "coalesce(edge.mention_key, '')",
        "edge.mention_key AS mention_key, edge.provenance_key AS provenance_key, "
        "source.opaque_key AS entity_key, target.opaque_key AS chunk_key, "
        "edge.document_key AS document_key, edge.chunk_number AS chunk_number, "
        "edge.confidence AS confidence",
    ),
)


def _legacy_query(relationship, cursor, fields):
    """Frozen pre-optimization query, independent of the production builder."""
    return (
        f"MATCH (source)-[edge:{relationship}]->(target) "
        "WHERE (source.generation_key = $generation_key"
        " OR target.generation_key = $generation_key"
        " OR edge.generation_key = $generation_key) "
        f"AND (NOT $has_cursor OR {cursor} > $cursor_key "
        f"OR ({cursor} = $cursor_key AND id(edge) > $cursor_id)) "
        "RETURN edge.generation_key AS generation_key, "
        "source.generation_key AS source_generation_key, "
        "target.generation_key AS target_generation_key, "
        f"{fields}, {cursor} AS cursor_key, id(edge) AS cursor_id "
        "ORDER BY cursor_key, cursor_id LIMIT $page_limit"
    )


@pytest.mark.container
def test_edge_pages_preserve_every_authority_arm_and_limit_before_wide_projection(
    isolated_memgraph_container,
    monkeypatch,
):
    target = isolated_memgraph_container
    driver = Neo4jMemgraphDriver(target["uri"], "", "", database=target["database"])
    monkeypatch.setattr(edges, "PAGE_SIZE", 2)
    # No endpoint labels: malformed/family-only nodes must remain visible. Tied
    # keys exercise id(edge) across page boundaries, including duplicate edges.
    cases = (
        ("g", "g", "g"),
        ("g", "other", "other"),
        ("other", "g", "other"),
        ("other", "other", "g"),
        ("g", None, None),
        (None, "g", None),
        (None, None, "g"),
        ("g", "g", "g"),
        ("other", "other", "other"),
    )
    try:
        with driver._connection().session() as session:
            for relationship, _cursor, _fields in _SPECS:
                session.run(f"CREATE EDGE INDEX ON :{relationship}").consume()
                for source_gen, target_gen, edge_gen in cases:
                    session.run(
                        "CREATE (source {generation_key:$sg,opaque_key:'tie'})"
                        f"-[edge:{relationship} {{generation_key:$eg,"
                        "mention_key:'tie',relation_key:'tie',evidence_key:'tie',"
                        "chunk_number:0,confidence:0.9}]->"
                        "(target {generation_key:$tg,opaque_key:'tie'})",
                        sg=source_gen,
                        tg=target_gen,
                        eg=edge_gen,
                    ).consume()
        parameters = {
            "generation_key": "g",
            "has_cursor": False,
            "cursor_key": "",
            "cursor_id": -1,
            "page_limit": 100,
        }
        for query, spec in zip(edges._EDGE_QUERIES, _SPECS, strict=True):
            baseline = driver.execute_read(
                _legacy_query(*spec),
                parameters,
                timeout_seconds=0.3,
                max_records=100,
            )
            assert len(baseline) == 8
            actual = tuple(edges._stream_edge_family(driver, "g", query, 8, 100, 0.3))
            assert actual == tuple(edges._mapping(row) for row in baseline)
            assert {row["source_generation_key"] for row in actual} == {
                "g",
                "other",
                None,
            }
            plans = driver.execute_read(
                "EXPLAIN " + query,
                parameters,
                timeout_seconds=0.3,
                max_records=100,
            )
            operators = [row["QUERY PLAN"].strip() for row in plans]
            assert any("ScanAllByEdgeType " in row for row in operators)
            # EXPLAIN lists the output operator first: property projection must
            # occur after the bounded window rather than for every scanned edge.
            wide = next(
                i
                for i, row in enumerate(operators)
                if "Produce {generation_key," in row
            )
            limit = next(i for i, row in enumerate(operators) if row == "* Limit")
            assert wide < limit
    finally:
        driver._client.close()


@pytest.mark.container
def test_indexed_edge_attestation_keeps_checksum_and_rejects_corrupt_topology(
    isolated_memgraph_container,
):
    target = isolated_memgraph_container
    driver = Neo4jMemgraphDriver(target["uri"], "", "", database=target["database"])
    repository = MemgraphProjectionRepository(driver)
    bundle = _bundle()
    generation = bundle.generation.generation_key
    expected = topology_edge_attestation(bundle)

    def observed():
        return edges.validated_topology_attestation(
            driver, generation, timeout_seconds=0.3
        )

    def assert_invalid():
        try:
            result = observed()
        except (KeyError, TypeError, ValueError):
            return
        assert result is None

    try:
        repository.write_staging_generation(
            bundle=bundle,
            private_mapping_checksum="d" * 64,
            batch_size=128,
            timeout_seconds=0.3,
        )
        assert observed() == expected
        with driver._connection().session() as session:
            for relationship, _cursor, _fields in _SPECS:
                row = (
                    session.run(
                        f"MATCH (s)-[e:{relationship}]->(t) RETURN id(e) AS edge_id, "
                        "id(s) AS source_id, id(t) AS target_id, "
                        "properties(e) AS properties "
                        "LIMIT 1",
                    )
                    .single()
                    .data()
                )
                # Missing/wrong edge generation remains in scope through its
                # endpoints, and must fail the immutable topology attestation.
                for corrupted in (None, "f" * 64):
                    session.run(
                        "MATCH ()-[e]->() WHERE id(e)=$edge_id "
                        "SET e.generation_key=$value",
                        edge_id=row["edge_id"],
                        value=corrupted,
                    ).consume()
                    assert_invalid()
                session.run(
                    "MATCH ()-[e]->() WHERE id(e)=$edge_id SET e.generation_key=$value",
                    edge_id=row["edge_id"],
                    value=generation,
                ).consume()
                # An identical extra physical relationship must not collapse
                # during indexed enumeration or hide behind tied cursors.
                duplicate = session.run(
                    "MATCH (s),(t) WHERE id(s)=$source_id AND id(t)=$target_id "
                    f"CREATE (s)-[e:{relationship}]->(t) SET e=$properties "
                    "RETURN id(e) AS edge_id",
                    source_id=row["source_id"],
                    target_id=row["target_id"],
                    properties=row["properties"],
                ).single()["edge_id"]
                assert_invalid()
                session.run(
                    "MATCH ()-[e]->() WHERE id(e)=$edge_id DELETE e",
                    edge_id=duplicate,
                ).consume()
                assert observed() == expected
                session.run(
                    "MATCH ()-[e]->() WHERE id(e)=$edge_id DELETE e",
                    edge_id=row["edge_id"],
                ).consume()
                assert_invalid()
                write_topology_edges(
                    driver, bundle, batch_size=128, timeout_seconds=0.3
                )
                assert observed() == expected
    finally:
        driver._client.close()
