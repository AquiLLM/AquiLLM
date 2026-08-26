from __future__ import annotations

import uuid
from dataclasses import asdict
from hashlib import sha256

import pytest

from apps.knowledge_graph.projection.memgraph_driver import (
    MemgraphWriteSummaryV1,
)
from apps.knowledge_graph.projection.memgraph_edges import (
    EDGE_FAMILIES,
    EDGE_ORDER_FIELDS,
    topology_edge_attestation,
    topology_edge_rows,
)
from apps.knowledge_graph.projection.memgraph_records import _dto
from apps.knowledge_graph.projection.memgraph_repository import (
    MemgraphProjectionRepository,
)
from apps.knowledge_graph.projection.records import (
    AutomaticCanonicalMembershipV1,
    CollectionGraphProjectionBundleV1,
    ProjectionGenerationManifestV1,
    ProjectionLifecycleState,
)
from apps.knowledge_graph.projection.serialization import projection_checksum
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
        if not self.read_results and "RETURN g AS marker" in cypher and self.writes:
            return ({"marker": dict(self.writes[0][1])},)
        return self.read_results.pop(0) if self.read_results else ()


def _bundle():
    return _BundleSource().load_projection_rows(
        projection_id=uuid.uuid4(), batch_size=10
    )


def test_memgraph_decoder_restores_omitted_optional_null_properties() -> None:
    decoded = _dto(
        AutomaticCanonicalMembershipV1,
        {
            "entity_key": "a" * 64,
            "decision_checksum": "b" * 64,
            "resolver_version": "resolver-v1",
            "resolution_config_checksum": "c" * 64,
        },
    )

    assert decoded.automatic_membership_key is None


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


def _family_rows(bundle, records, identity):
    return tuple(
        {
            "record": {
                **asdict(row),
                "opaque_key": getattr(row, identity),
                "generation_key": bundle.generation.generation_key,
            }
        }
        for row in records
    )


def _record_reads(bundle):
    return [
        ({"record": {**asdict(bundle.generation), "graph_checksum": "ignored"}},),
        _family_rows(bundle, bundle.entities, "entity_key"),
        _family_rows(bundle, bundle.automatic_memberships, "entity_key"),
        _family_rows(bundle, bundle.documents, "document_key"),
        _family_rows(bundle, bundle.chunks, "chunk_key"),
        _family_rows(bundle, bundle.relation_semantics, "semantics_key"),
        _family_rows(bundle, bundle.relations, "relation_key"),
        _family_rows(bundle, bundle.evidence, "evidence_key"),
        _family_rows(bundle, bundle.entity_mentions, "mention_key"),
        _family_rows(bundle, bundle.artifact_provenance, "scope_key"),
    ]


def _stream_record_reads(bundle):
    rows = _record_reads(bundle)
    return [
        rows[0],
        rows[9],
        rows[2],
        rows[4],
        rows[3],
        rows[1],
        rows[8],
        rows[7],
        rows[5],
        rows[6],
    ]


def _topology_reads(bundle):
    attestation = topology_edge_attestation(bundle)
    counts = {
        f"{family}_count": count
        for family, count in zip(EDGE_FAMILIES, attestation.counts, strict=True)
    }
    marker = {"topology_checksum": attestation.checksum, **counts}
    return [
        (marker,),
        *(
            tuple(
                {"edge": row}
                for row in sorted(family, key=lambda item: item[order_field])
            )
            for family, order_field in zip(
                topology_edge_rows(bundle), EDGE_ORDER_FIELDS, strict=True
            )
        ),
    ]


def test_repository_uses_parameterized_idempotent_staging_and_ready_last() -> None:
    driver = _FakeDriver()
    repository = MemgraphProjectionRepository(driver)
    bundle = _closed_bundle()

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
    assert any("ProjectedRelationSemantics" in call[0] for call in driver.writes)
    assert any("ProjectedEntityMention" in call[0] for call in driver.writes)
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
    driver.read_results.extend(_stream_record_reads(bundle))
    driver.read_results.extend(_topology_reads(bundle))
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
    ready_state["topology_checksum"] = topology_edge_attestation(bundle).checksum
    ready_state["validated_topology_checksum"] = ready_state["topology_checksum"]
    driver.read_results.append((ready_state,))
    driver.read_results.extend(_topology_reads(bundle))
    driver.read_results.append(({**ready_state, "state": "ready"},))
    reads_before_ready = len(driver.reads)

    repository.mark_generation_ready(
        generation_key=repository.opaque_generation_key("1" * 64),
        validation_checksum=validation.validation_checksum,
        timeout_seconds=1.0,
    )
    assert validation.valid is True
    assert driver.writes[-2][1]["validation_checksum"] == expected.graph_checksum
    assert driver.writes[-2][1]["validated_topology_checksum"] == (
        topology_edge_attestation(bundle).checksum
    )
    assert driver.writes[-1][1]["state"] == "ready"
    assert "MATCH" in driver.writes[-1][0]
    assert any(
        "PROJECTED_RELATION" in read[0]
        for read in driver.reads[reads_before_ready:]
    )
    ready_reads = [read[0] for read in driver.reads[reads_before_ready:]]
    assert any(
        "DOCUMENT_CHUNK" in query and "ORDER BY cursor_key, cursor_id" in query
        for query in ready_reads
    )
    assert any(
        "ENTITY_MENTION" in query and "ORDER BY cursor_key, cursor_id" in query
        for query in ready_reads
    )


def test_validation_rejects_count_or_endpoint_drift_without_publishing_token() -> None:
    driver = _FakeDriver()
    repository = MemgraphProjectionRepository(driver)
    bundle = CollectionGraphProjectionBundleV1(**_bundle())
    expected = _expected_manifest(bundle)
    driver.read_results.append((_manifest_row(expected),))
    records = _stream_record_reads(bundle)
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


@pytest.mark.parametrize(("node_count", "expected"), ((0, False), (3, True)))
def test_generation_deletion_reports_attested_node_count(node_count, expected) -> None:
    driver = _FakeDriver()
    driver.execute_write = lambda *_args, **_kwargs: MemgraphWriteSummaryV1(
        {"nodes_deleted": node_count}
    )
    repository = MemgraphProjectionRepository(driver)

    deleted = repository.delete_generation(
        generation_key=repository.opaque_generation_key("a" * 64),
        timeout_seconds=1.0,
    )

    assert deleted is expected


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
