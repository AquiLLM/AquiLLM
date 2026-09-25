from __future__ import annotations

from hashlib import sha256

import pytest

from apps.knowledge_graph.projection.memgraph_edge_validation import (
    _EDGE_QUERIES,
    validate_topology_marker,
)
from apps.knowledge_graph.projection.memgraph_edges import (
    EDGE_FAMILIES,
    topology_edge_attestation_from_rows,
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
    _stream_record_reads,
    _topology_reads,
)
from apps.knowledge_graph.tests.test_projection_records import _bundle as _closed_bundle


def test_validation_rereads_exact_dtos_and_detects_property_corruption() -> None:
    bundle = _closed_bundle()
    expected = ProjectionGenerationManifestV1(
        bundle.generation.generation_key,
        bundle.generation.schema_version,
        bundle.generation.projection_version,
        bundle.generation.identifier_key_version,
        projection_checksum(bundle),
        projection_checksum(bundle),
        "d" * 64,
        bundle.counts,
        ProjectionLifecycleState.BUILDING,
    )
    staged = _manifest_row(expected)
    staged["private_mapping_checksum"] = sha256(b"[]").hexdigest()
    staged["state"] = "staging"
    driver = _FakeDriver()
    driver.read_results.append((staged,))
    records = _stream_record_reads(bundle)
    corrupted = dict(records[1][0]["record"])
    corrupted["retrieval_utility"] = 0.125
    records[1] = ({"record": corrupted}, *records[1][1:])
    driver.read_results.extend(records)

    validation = MemgraphProjectionRepository(driver).validate_generation(
        expected=expected, timeout_seconds=1.0
    )

    assert validation.valid is False
    assert driver.writes == []


def test_nonempty_generation_rejects_empty_private_mapping_checksum() -> None:
    bundle = _closed_bundle()
    empty = sha256(b"[]").hexdigest()
    expected = ProjectionGenerationManifestV1(
        bundle.generation.generation_key,
        bundle.generation.schema_version,
        bundle.generation.projection_version,
        bundle.generation.identifier_key_version,
        projection_checksum(bundle),
        projection_checksum(bundle),
        empty,
        bundle.counts,
        ProjectionLifecycleState.BUILDING,
    )
    driver = _FakeDriver()
    driver.read_results.append((_manifest_row(expected),))
    driver.read_results.extend(_stream_record_reads(bundle))

    validation = MemgraphProjectionRepository(driver).validate_generation(
        expected=expected, timeout_seconds=1.0
    )

    assert validation.valid is False
    assert driver.writes == []


def test_ready_rejects_missing_private_mapping_checksum() -> None:
    driver = _FakeDriver()
    driver.read_results.append(
        (
            {
                "validation_checksum": "a" * 64,
                "graph_checksum": "a" * 64,
                "private_mapping_checksum": None,
                "chunk_count": 1,
                "state": "building",
            },
        )
    )

    with pytest.raises(ValueError, match="validation checksum/state mismatch"):
        MemgraphProjectionRepository(driver).mark_generation_ready(
            generation_key=MemgraphProjectionRepository.opaque_generation_key("1" * 64),
            validation_checksum="a" * 64,
            timeout_seconds=1.0,
        )

    assert driver.writes == []


def test_ready_rejects_private_checksum_changed_after_validation() -> None:
    driver = _FakeDriver()
    driver.read_results.append(
        (
            {
                "validation_checksum": "a" * 64,
                "graph_checksum": "a" * 64,
                "private_mapping_checksum": "b" * 64,
                "validated_private_mapping_checksum": "c" * 64,
                "chunk_count": 1,
                "state": "building",
            },
        )
    )

    with pytest.raises(ValueError, match="validation checksum/state mismatch"):
        MemgraphProjectionRepository(driver).mark_generation_ready(
            generation_key=MemgraphProjectionRepository.opaque_generation_key("1" * 64),
            validation_checksum="a" * 64,
            timeout_seconds=1.0,
        )

    assert driver.writes == []


def test_ready_fails_closed_when_publication_compare_and_set_is_lost() -> None:
    empty_topology = topology_edge_attestation_from_rows(((), (), (), (), ()))
    validated = {
        "validation_checksum": "a" * 64,
        "graph_checksum": "a" * 64,
        "private_mapping_checksum": "b" * 64,
        "validated_private_mapping_checksum": "b" * 64,
        "chunk_count": 1,
        "topology_checksum": empty_topology.checksum,
        "validated_topology_checksum": empty_topology.checksum,
        "state": "building",
    }
    driver = _FakeDriver()
    driver.read_results.append((validated,))
    driver.read_results.append(
        (
            {
                "topology_checksum": empty_topology.checksum,
                **{f"{family}_count": 0 for family in EDGE_FAMILIES},
            },
        )
    )
    driver.read_results.extend(((), (), (), (), ()))
    driver.read_results.append((validated,))

    with pytest.raises(ValueError, match="publication fence was lost"):
        MemgraphProjectionRepository(driver).mark_generation_ready(
            generation_key=MemgraphProjectionRepository.opaque_generation_key("1" * 64),
            validation_checksum="a" * 64,
            timeout_seconds=1.0,
        )


def test_edge_attestation_reads_foreign_and_missing_generation_relationships() -> None:
    assert _EDGE_QUERIES
    assert all(
        "edge.generation_key:$generation_key" not in query for query in _EDGE_QUERIES
    )
    assert all(
        "source.generation_key = $generation_key"
        " OR target.generation_key = $generation_key" in query
        for query in _EDGE_QUERIES
    )
    assert all(
        "OR edge.generation_key = $generation_key" in query
        for query in _EDGE_QUERIES
    )


def test_edge_attestation_rejects_wrong_generation_relationship() -> None:
    bundle = _closed_bundle()
    reads = _topology_reads(bundle)
    membership_rows = list(reads[1])
    membership_rows[0] = {
        "edge": {**membership_rows[0]["edge"], "generation_key": "f" * 64}
    }
    driver = _FakeDriver()
    driver.read_results = [reads[0], tuple(membership_rows), *reads[2:]]

    assert (
        validate_topology_marker(
            driver,
            bundle.generation.generation_key,
            timeout_seconds=1.0,
        )
        is False
    )
