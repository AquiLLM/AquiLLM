from __future__ import annotations

from hashlib import sha256

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


def _expected(state: ProjectionLifecycleState):
    bundle = _bundle()
    checksum = projection_checksum(bundle)
    return bundle, ProjectionGenerationManifestV1(
        bundle.generation.generation_key,
        bundle.generation.schema_version,
        bundle.generation.projection_version,
        bundle.generation.identifier_key_version,
        checksum,
        checksum,
        "d" * 64,
        bundle.counts,
        state,
    )


def test_validation_rejects_a_staged_private_mapping_checksum_mismatch() -> None:
    driver = _FakeDriver()
    repository = MemgraphProjectionRepository(driver)
    bundle, expected = _expected(ProjectionLifecycleState.BUILDING)
    staged = _manifest_row(expected)
    staged["private_mapping_checksum"] = sha256(b"[]").hexdigest()
    staged["state"] = "staging"
    driver.read_results.append((staged,))
    driver.read_results.extend(_record_reads(bundle))

    validation = repository.validate_generation(
        expected=expected,
        timeout_seconds=1.0,
    )

    assert validation.valid is False
    assert driver.writes == []


def test_ready_validation_rereads_records_without_republishing_marker() -> None:
    driver = _FakeDriver()
    repository = MemgraphProjectionRepository(driver)
    bundle, expected = _expected(ProjectionLifecycleState.READY)
    driver.read_results.append((_manifest_row(expected),))
    driver.read_results.extend(_record_reads(bundle))

    validation = repository.validate_generation(
        expected=expected,
        timeout_seconds=1.0,
    )

    assert validation.valid is True
    assert validation.validation_checksum == expected.graph_checksum
    assert driver.writes == []
