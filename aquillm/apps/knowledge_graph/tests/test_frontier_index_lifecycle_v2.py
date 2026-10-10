# ruff: noqa: F401, F811
"""Legacy ready generations are never upgraded in place; authority access is bounded."""

from dataclasses import replace
from uuid import uuid4

import pytest

from apps.knowledge_graph.projection.frontier_cypher_v2 import MANIFEST_LOOKUP
from apps.knowledge_graph.projection.frontier_index_contracts_v2 import MARKER_FIELDS
from apps.knowledge_graph.projection.frontier_index_store_v2 import read_marker
from apps.knowledge_graph.projection.memgraph_repository import (
    MemgraphProjectionRepository,
)
from apps.knowledge_graph.tests.frontier_fixture_support import bridge_bundles
from apps.knowledge_graph.tests.test_canonical_frontier_memgraph import (
    frontier_memgraph,
)
from apps.knowledge_graph.tests.test_memgraph_projection_repository import (
    _expected_manifest,
)


def test_existing_legacy_ready_generation_rejects_in_place_index_upgrade(
    frontier_memgraph,
):
    repository = MemgraphProjectionRepository(frontier_memgraph)
    bundle = bridge_bundles()[0]
    generation = repository.opaque_generation_key(bundle.generation.generation_key)
    repository.delete_generation(generation_key=generation, timeout_seconds=5.0)
    try:
        expected = replace(
            _expected_manifest(bundle), private_mapping_checksum="d" * 64
        )
        repository.write_staging_generation(
            bundle=bundle,
            private_mapping_checksum="d" * 64,
            batch_size=128,
            timeout_seconds=5.0,
        )
        validation = repository.validate_generation(
            expected=expected, timeout_seconds=5.0
        )
        assert validation.valid
        repository.mark_generation_ready(
            generation_key=generation,
            validation_checksum=validation.validation_checksum,
            timeout_seconds=5.0,
        )
        before = read_marker(frontier_memgraph, generation.value, 5.0)
        assert not set(MARKER_FIELDS) & set(before)
        with pytest.raises(ValueError, match="fence"):
            repository.write_staging_generation(
                bundle=bundle,
                private_mapping_checksum="d" * 64,
                batch_size=128,
                timeout_seconds=5.0,
                frontier_index=True,
            )
        assert read_marker(frontier_memgraph, generation.value, 5.0) == before
    finally:
        repository.delete_generation(generation_key=generation, timeout_seconds=5.0)


def test_manifest_equality_scan_stops_at_two_duplicate_records(frontier_memgraph):
    driver, generation = frontier_memgraph, uuid4().hex
    driver.ensure_projection_schema(timeout_seconds=5.0)
    driver.execute_write(
        "UNWIND range(1,201) AS i CREATE (:CollectionGeneration {generation_key:$key})",
        {"key": generation},
        timeout_seconds=5.0,
    )
    try:
        params = {"generation_key": generation}
        plan = driver.execute_read(
            "EXPLAIN " + MANIFEST_LOOKUP, params, timeout_seconds=5.0, max_records=100
        )
        with driver._connection().session(database="memgraph") as session:
            profile = tuple(session.run("PROFILE " + MANIFEST_LOOKUP, params).data())
        print("FRONTIER_MANIFEST_EXPLAIN", plan)
        print("FRONTIER_MANIFEST_PROFILE", profile)
        scans = [r for r in profile if "ScanAllByLabelProperties" in r["OPERATOR"]]
        assert scans and all(r["ACTUAL HITS"] == 2 for r in scans)
        assert not any("OrderBy" in row["QUERY PLAN"] for row in plan)
        assert (
            len(
                driver.execute_read(
                    MANIFEST_LOOKUP, params, timeout_seconds=5.0, max_records=2
                )
            )
            == 2
        )
    finally:
        driver.execute_write(
            "MATCH (n:CollectionGeneration {generation_key:$key}) DELETE n",
            {"key": generation},
            timeout_seconds=5.0,
        )
