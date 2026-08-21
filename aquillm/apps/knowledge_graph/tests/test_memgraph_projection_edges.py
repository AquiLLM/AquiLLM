from __future__ import annotations

from apps.knowledge_graph.projection.memgraph_edges import (
    EDGE_FAMILIES,
    topology_edge_attestation,
    topology_edge_rows,
)
from apps.knowledge_graph.projection.memgraph_repository import (
    MemgraphProjectionRepository,
)
from apps.knowledge_graph.projection.records import (
    ProjectionGenerationManifestV1,
    ProjectionLifecycleState,
)
from apps.knowledge_graph.projection.serialization import projection_checksum
from apps.knowledge_graph.tests.test_memgraph_projection_repository import (
    _FakeDriver,
    _manifest_row,
    _record_reads,
)
from apps.knowledge_graph.tests.test_projection_records import _bundle


def test_repository_projects_generation_scoped_topology_edges() -> None:
    driver = _FakeDriver()
    bundle = _bundle()
    MemgraphProjectionRepository(driver).write_staging_generation(
        bundle=bundle,
        private_mapping_checksum="d" * 64,
        batch_size=10,
        timeout_seconds=1.0,
    )

    cypher = "\n".join(call[0] for call in driver.writes)
    assert "PROJECTED_RELATION" in cypher
    assert "ENTITY_MENTION" in cypher
    assert "RELATION_EVIDENCE" in cypher
    assert "DOCUMENT_CHUNK" in cypher
    assert all(
        "generation_key:$generation_key" in statement
        for statement, _parameters, _timeout in driver.writes
        if any(
            label in statement
            for label in (
                "PROJECTED_RELATION",
                "ENTITY_MENTION",
                "RELATION_EVIDENCE",
                "DOCUMENT_CHUNK",
            )
        )
    )


def test_staging_marker_binds_derived_topology_counts_and_checksum() -> None:
    driver = _FakeDriver()
    bundle = _bundle()
    expected = topology_edge_attestation(bundle)
    MemgraphProjectionRepository(driver).write_staging_generation(
        bundle=bundle,
        private_mapping_checksum="d" * 64,
        batch_size=10,
        timeout_seconds=1.0,
    )

    marker = driver.writes[0][1]
    assert marker["topology_checksum"] == expected.checksum
    counts = tuple(marker[f"{family}_count"] for family in EDGE_FAMILIES)
    assert counts == expected.counts


def test_validation_rejects_missing_physical_topology_edge() -> None:
    driver = _FakeDriver()
    repository = MemgraphProjectionRepository(driver)
    bundle = _bundle()
    checksum = projection_checksum(bundle)
    expected = ProjectionGenerationManifestV1(
        bundle.generation.generation_key,
        bundle.generation.schema_version,
        bundle.generation.projection_version,
        bundle.generation.identifier_key_version,
        checksum,
        checksum,
        "d" * 64,
        bundle.counts,
        ProjectionLifecycleState.BUILDING,
    )
    attestation = topology_edge_attestation(bundle)
    staged = _manifest_row(expected)
    staged["state"] = "staging"
    driver.read_results.append((staged,))
    driver.read_results.extend(_record_reads(bundle))
    driver.read_results.append(
        (
            {
                "topology_checksum": attestation.checksum,
                **{
                    f"{family}_count": count
                    for family, count in zip(
                        EDGE_FAMILIES, attestation.counts, strict=True
                    )
                },
            },
        )
    )
    edge_reads = [
        tuple({"edge": row} for row in rows) for rows in topology_edge_rows(bundle)
    ]
    edge_reads[EDGE_FAMILIES.index("projected_relation")] = ()
    driver.read_results.extend(edge_reads)

    validation = repository.validate_generation(
        expected=expected,
        timeout_seconds=1.0,
    )

    assert validation.valid is False
    assert driver.writes == []
