from __future__ import annotations

from hashlib import sha256

import pytest

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
    records = _record_reads(bundle)
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
    driver.read_results.extend(_record_reads(bundle))

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
    validated = {
        "validation_checksum": "a" * 64,
        "graph_checksum": "a" * 64,
        "private_mapping_checksum": "b" * 64,
        "validated_private_mapping_checksum": "b" * 64,
        "chunk_count": 1,
        "state": "building",
    }
    driver = _FakeDriver()
    driver.read_results.extend(((validated,), (validated,)))

    with pytest.raises(ValueError, match="publication fence was lost"):
        MemgraphProjectionRepository(driver).mark_generation_ready(
            generation_key=MemgraphProjectionRepository.opaque_generation_key("1" * 64),
            validation_checksum="a" * 64,
            timeout_seconds=1.0,
        )
