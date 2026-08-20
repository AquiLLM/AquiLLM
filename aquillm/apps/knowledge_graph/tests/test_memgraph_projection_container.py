from __future__ import annotations

import pytest

from apps.knowledge_graph.projection.memgraph_driver import Neo4jMemgraphDriver
from apps.knowledge_graph.projection.memgraph_repository import (
    MemgraphProjectionRepository,
)
from apps.knowledge_graph.projection.records import (
    ProjectionGenerationManifestV1,
    ProjectionLifecycleState,
)
from apps.knowledge_graph.projection.serialization import projection_checksum
from apps.knowledge_graph.retrieval.topology.contracts import (
    HybridBranchKind,
    TopologyCapsV1,
)
from apps.knowledge_graph.tests.memgraph_test_support import (
    isolated_memgraph_container as _isolated_memgraph_container,
)
from apps.knowledge_graph.tests.test_projection_records import _bundle as _closed_bundle


@pytest.fixture
def isolated_memgraph_container():
    yield from _isolated_memgraph_container.__wrapped__()


@pytest.mark.container
def test_memgraph_repository_stage_read_validate_ready_delete_end_to_end(
    isolated_memgraph_container,
) -> None:
    target = isolated_memgraph_container
    driver = Neo4jMemgraphDriver(target["uri"], "", "", database=target["database"])
    repository = MemgraphProjectionRepository(driver)
    repository.ensure_schema(timeout_seconds=5.0)
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
    key = repository.opaque_generation_key(bundle.generation.generation_key)
    try:
        repository.write_staging_generation(
            bundle=bundle, batch_size=2, timeout_seconds=5.0
        )
        assert (
            repository.read_generation_records(
                generation_key=key,
                caps=TopologyCapsV1(HybridBranchKind.DIRECT, 2, 1, 10, 10, 2),
                timeout_seconds=5.0,
            )
            == bundle
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
        assert (
            repository.read_generation_manifest(
                generation_key=key, timeout_seconds=5.0
            ).state
            is ProjectionLifecycleState.READY
        )
    finally:
        repository.delete_generation(generation_key=key, timeout_seconds=5.0)
        with pytest.raises(ValueError, match="manifest is missing"):
            repository.read_generation_manifest(generation_key=key, timeout_seconds=5.0)
