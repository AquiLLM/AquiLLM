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
from apps.knowledge_graph.tests.test_memgraph_projection_batches import _large_bundle
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
            bundle=bundle,
            private_mapping_checksum=expected.private_mapping_checksum,
            batch_size=2,
            timeout_seconds=5.0,
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


@pytest.mark.container
def test_batched_projection_roundtrip_retry_indexes_and_rollback_at_live_timeout(
    isolated_memgraph_container,
):
    from dataclasses import asdict
    from time import perf_counter

    from neo4j import GraphDatabase

    from apps.knowledge_graph.projection.memgraph_driver import MemgraphDriverError
    from apps.knowledge_graph.projection.memgraph_edges import (
        write_parameterized_batches,
        write_topology_edges,
    )

    target = isolated_memgraph_container

    class TimedDriver(Neo4jMemgraphDriver):
        def __init__(self):
            super().__init__(target["uri"], "", "", database=target["database"])
            self.writes = []

        def execute_write(self, query, parameters, *, timeout_seconds):
            started = perf_counter()
            result = super().execute_write(
                query, parameters, timeout_seconds=timeout_seconds
            )
            self.writes.append((query, parameters, perf_counter() - started))
            return result

    driver = TimedDriver()
    repository = MemgraphProjectionRepository(driver)
    bundle = _large_bundle()
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
    key = repository.opaque_generation_key(bundle.generation.generation_key)
    timeout = 0.3

    # No manual schema bootstrap: normal writes must create all required indexes.
    # A retry replays the complete bundle and existing DDL without duplicates.
    for _ in range(2):
        repository.write_staging_generation(
            bundle=bundle,
            private_mapping_checksum="d" * 64,
            batch_size=500,
            timeout_seconds=timeout,
        )
    assert (
        repository.read_generation_records(
            generation_key=key,
            caps=TopologyCapsV1(HybridBranchKind.DIRECT, 2, 1, 10, 300, 2),
            timeout_seconds=timeout,
        )
        == bundle
    )
    assert repository.validate_generation(
        expected=expected, timeout_seconds=timeout
    ).valid

    child_queries = {
        query: params for query, params, _ in driver.writes if "UNWIND" in query
    }
    for query, params in child_queries.items():
        plan = driver.execute_read(
            "EXPLAIN " + query, params, timeout_seconds=timeout, max_records=100
        )
        operators = "\n".join(row["QUERY PLAN"] for row in plan)
        assert "ScanAll (" not in operators
        assert ":ProjectedRecord {generation_key, opaque_key}" in operators

    # A missing endpoint cannot produce a publishable partial batch.
    driver.execute_write(
        "MATCH (c:ProjectedChunk:ProjectedRecord "
        "{generation_key:$generation_key}) DETACH DELETE c",
        {"generation_key": key.value},
        timeout_seconds=timeout,
    )
    write_topology_edges(driver, bundle, batch_size=128, timeout_seconds=timeout)
    assert not repository.validate_generation(
        expected=expected, timeout_seconds=timeout
    ).valid
    repository.write_staging_generation(
        bundle=bundle,
        private_mapping_checksum="d" * 64,
        batch_size=500,
        timeout_seconds=timeout,
    )
    validation = repository.validate_generation(
        expected=expected, timeout_seconds=timeout
    )
    assert validation.valid
    repository.mark_generation_ready(
        generation_key=key,
        validation_checksum=validation.validation_checksum,
        timeout_seconds=timeout,
    )
    with pytest.raises(ValueError, match="staging generation fence"):
        repository.write_staging_generation(
            bundle=bundle,
            private_mapping_checksum="d" * 64,
            batch_size=500,
            timeout_seconds=timeout,
        )

    # Driver-managed transactions roll back a whole batch if commit validation
    # fails, and a deterministic MERGE retry remains idempotent for duplicate rows.
    rollback_key = "e" * 64
    driver.execute_write(
        "CREATE (:CollectionGeneration "
        "{generation_key:$generation_key, state:'staging'})",
        {"generation_key": rollback_key},
        timeout_seconds=timeout,
    )
    with GraphDatabase.driver(target["uri"], auth=None) as raw:
        with raw.session(database=target["database"]) as session:
            session.run(
                "CREATE CONSTRAINT ON (n:BatchRollback) ASSERT n.key IS UNIQUE"
            ).consume()
    options = dict(generation_key=rollback_key, batch_size=2, timeout_seconds=timeout)
    duplicate_rows = [{"key": "duplicate"}, {"key": "duplicate"}]
    with pytest.raises(MemgraphDriverError, match="memgraph_write_failed"):
        write_parameterized_batches(
            driver, "CREATE (n:BatchRollback {key:row.key})", duplicate_rows, **options
        )
    assert driver.execute_read(
        "MATCH (n:BatchRollback) RETURN count(n) AS count",
        {},
        timeout_seconds=timeout,
        max_records=1,
    ) == ({"count": 0},)
    for _ in range(2):
        write_parameterized_batches(
            driver, "MERGE (n:BatchRollback {key:row.key})", duplicate_rows, **options
        )
    assert driver.execute_read(
        "MATCH (n:BatchRollback) RETURN count(n) AS count",
        {},
        timeout_seconds=timeout,
        max_records=1,
    ) == ({"count": 1},)

    # Exercise the widest real record schema at the internal maximum, including
    # cold compilation and the deployed 300ms transaction timeout.
    rows = [
        {**asdict(bundle.artifact_provenance[0]), "opaque_key": f"{index:064x}"}
        for index in range(128)
    ]
    assignments = ", ".join(f"n.{name}=row.{name}" for name in sorted(rows[0]))
    write_parameterized_batches(
        driver,
        "MERGE (n:BatchWide:ProjectedRecord "
        "{generation_key:$generation_key, opaque_key:row.opaque_key}) SET "
        + assignments,
        rows,
        generation_key=rollback_key,
        batch_size=500,
        timeout_seconds=timeout,
    )
    assert driver.execute_read(
        "MATCH (n:BatchWide) RETURN count(n) AS count",
        {},
        timeout_seconds=timeout,
        max_records=1,
    ) == ({"count": 128},)
    print(
        {
            "max_batch_wall_ms": round(
                max(elapsed for query, _, elapsed in driver.writes if "UNWIND" in query)
                * 1000,
                2,
            ),
            "wide_batch_wall_ms": round(driver.writes[-1][2] * 1000, 2),
        }
    )
