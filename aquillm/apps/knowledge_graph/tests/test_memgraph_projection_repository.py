from __future__ import annotations

import uuid
from dataclasses import asdict
from hashlib import sha256

from apps.knowledge_graph.projection.memgraph_driver import (
    MemgraphWriteSummaryV1,
)
from apps.knowledge_graph.projection.memgraph_repository import (
    MemgraphProjectionRepository,
)
from apps.knowledge_graph.projection.records import (
    CollectionGraphProjectionBundleV1,
    ProjectionGenerationManifestV1,
    ProjectionLifecycleState,
)
from apps.knowledge_graph.projection.serialization import projection_checksum
from apps.knowledge_graph.retrieval.topology.contracts import (
    HybridBranchKind,
    TopologyCapsV1,
)
from apps.knowledge_graph.tests.test_projection_postgres_repository import _BundleSource
from apps.knowledge_graph.tests.test_projection_records import _bundle as _closed_bundle


class _FakeDriver:
    def __init__(self) -> None:
        self.writes = []
        self.reads = []
        self.read_results = []

    def execute_write(self, cypher, parameters, *, timeout_seconds):
        self.writes.append((cypher, parameters, timeout_seconds))
        return MemgraphWriteSummaryV1({"nodes_created": 0})

    def execute_read(self, cypher, parameters, *, timeout_seconds, max_records):
        self.reads.append((cypher, parameters, timeout_seconds, max_records))
        return self.read_results.pop(0) if self.read_results else ()


def _bundle():
    return _BundleSource().load_projection_rows(
        projection_id=uuid.uuid4(), batch_size=10
    )


def _expected_manifest(bundle):
    checksum = projection_checksum(bundle)
    return ProjectionGenerationManifestV1(
        bundle.generation.generation_key,
        bundle.generation.schema_version,
        bundle.generation.projection_version,
        bundle.generation.identifier_key_version,
        checksum,
        checksum,
        sha256(b"[]").hexdigest(),
        bundle.counts,
        ProjectionLifecycleState.BUILDING,
    )


def _manifest_row(manifest):
    row = asdict(manifest)
    row.update(row.pop("counts"))
    row["state"] = manifest.state.value
    return row


def _record_reads(bundle):
    return [
        ({"record": {**asdict(bundle.generation), "graph_checksum": "ignored"}},),
        tuple(
            {"record": {**asdict(row), "opaque_key": row.entity_key}}
            for row in bundle.entities
        ),
        tuple(
            {"record": {**asdict(row), "opaque_key": row.entity_key}}
            for row in bundle.automatic_memberships
        ),
        tuple(
            {"record": {**asdict(row), "opaque_key": row.document_key}}
            for row in bundle.documents
        ),
        tuple(
            {
                "record": {
                    **asdict(row),
                    "opaque_key": row.chunk_key,
                    "generation_key": bundle.generation.generation_key,
                }
            }
            for row in bundle.chunks
        ),
        tuple(
            {
                "record": {
                    **asdict(row),
                    "opaque_key": row.relation_key,
                    "generation_key": bundle.generation.generation_key,
                }
            }
            for row in bundle.relations
        ),
        tuple(
            {
                "record": {
                    **asdict(row),
                    "opaque_key": row.evidence_key,
                    "generation_key": bundle.generation.generation_key,
                }
            }
            for row in bundle.evidence
        ),
        tuple(
            {
                "record": {
                    **asdict(row),
                    "opaque_key": row.scope_key,
                    "generation_key": bundle.generation.generation_key,
                }
            }
            for row in bundle.artifact_provenance
        ),
    ]


def test_repository_uses_parameterized_idempotent_staging_and_ready_last() -> None:
    from apps.knowledge_graph.projection.records import (
        CollectionGraphProjectionBundleV1,
    )

    driver = _FakeDriver()
    repository = MemgraphProjectionRepository(driver)
    bundle = CollectionGraphProjectionBundleV1(**_bundle())

    private_checksum = "d" * 64
    repository.write_staging_generation(
        bundle=bundle,
        private_mapping_checksum=private_checksum,
        batch_size=2,
        timeout_seconds=1.0,
    )

    assert all("MERGE" in cypher for cypher, _parameters, _timeout in driver.writes)
    assert all(
        "1" * 64 not in cypher for cypher, _parameters, _timeout in driver.writes
    )
    assert driver.writes[0][1]["state"] == "staging"
    assert driver.writes[0][1]["private_mapping_checksum"] == private_checksum
    assert "g.private_mapping_checksum=$private_mapping_checksum" in driver.writes[0][0]
    assert not any(call[1].get("state") == "ready" for call in driver.writes)


def test_ready_marker_requires_matching_validated_manifest_and_is_last_write() -> None:
    driver = _FakeDriver()
    repository = MemgraphProjectionRepository(driver)
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
    staged["state"] = "staging"
    driver.read_results.append((staged,))
    driver.read_results.extend(_record_reads(bundle))
    driver.read_results.append((_manifest_row(expected),))

    validation = repository.validate_generation(
        expected=expected,
        timeout_seconds=1.0,
    )
    ready_state = {
        "validation_checksum": validation.validation_checksum,
        "graph_checksum": validation.validation_checksum,
        "private_mapping_checksum": expected.private_mapping_checksum,
        "validated_private_mapping_checksum": expected.private_mapping_checksum,
        "chunk_count": expected.counts.chunk_count,
        "state": "building",
    }
    driver.read_results.extend(((ready_state,), ({**ready_state, "state": "ready"},)))

    repository.mark_generation_ready(
        generation_key=repository.opaque_generation_key("1" * 64),
        validation_checksum=validation.validation_checksum,
        timeout_seconds=1.0,
    )

    assert validation.valid is True
    assert driver.writes[-2][1]["validation_checksum"] == expected.graph_checksum
    assert driver.writes[-1][1]["state"] == "ready"
    assert "MATCH" in driver.writes[-1][0]


def test_validation_rejects_count_or_endpoint_drift_without_publishing_token() -> None:
    driver = _FakeDriver()
    repository = MemgraphProjectionRepository(driver)
    bundle = CollectionGraphProjectionBundleV1(**_bundle())
    expected = _expected_manifest(bundle)
    driver.read_results.append((_manifest_row(expected),))
    records = _record_reads(bundle)
    records[1] = records[1] + records[1]
    driver.read_results.extend(records)

    validation = repository.validate_generation(
        expected=expected,
        timeout_seconds=1.0,
    )

    assert validation.valid is False
    assert driver.writes == []


def test_generation_deletion_is_generation_scoped_and_parameterized() -> None:
    driver = _FakeDriver()
    repository = MemgraphProjectionRepository(driver)
    key = repository.opaque_generation_key("a" * 64)

    repository.delete_generation(generation_key=key, timeout_seconds=1.0)

    cypher, parameters, _timeout = driver.writes[-1]
    assert "DETACH DELETE" in cypher
    assert "a" * 64 not in cypher
    assert parameters == {"generation_key": "a" * 64}


def test_generation_listing_supports_bounded_global_opaque_cursor_paging() -> None:
    driver = _FakeDriver()
    bundle = CollectionGraphProjectionBundleV1(**_bundle())
    expected = _expected_manifest(bundle)
    driver.read_results.append(({"manifest": _manifest_row(expected)},))
    repository = MemgraphProjectionRepository(driver)

    rows = repository.list_generations(
        collection_key=None,
        after_generation_key=repository.opaque_generation_key("0" * 64),
        limit=17,
        timeout_seconds=1.0,
    )

    assert rows == (expected,)
    cypher, parameters, _timeout, maximum = driver.reads[-1]
    assert "collection_key:$collection_key" not in cypher
    assert "g.generation_key > $after_generation_key" in cypher
    assert parameters == {"after_generation_key": "0" * 64, "limit": 17}
    assert maximum == 17


def test_generation_records_are_bounded_and_endpoint_closed() -> None:
    from apps.knowledge_graph.projection.records import (
        CollectionGraphProjectionBundleV1,
    )

    bundle = CollectionGraphProjectionBundleV1(**_bundle())
    driver = _FakeDriver()
    driver.read_results = _record_reads(bundle)
    caps = TopologyCapsV1(HybridBranchKind.DIRECT, 2, 1, 2, 2, 2)

    observed = MemgraphProjectionRepository(driver).read_generation_records(
        generation_key=MemgraphProjectionRepository.opaque_generation_key("1" * 64),
        caps=caps,
        timeout_seconds=1.0,
    )

    assert observed == bundle
    assert all(call[3] <= 2 for call in driver.reads)
