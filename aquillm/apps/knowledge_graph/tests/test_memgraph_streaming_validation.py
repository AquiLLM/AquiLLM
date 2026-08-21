from __future__ import annotations

from dataclasses import asdict, replace

import pytest

from apps.knowledge_graph.projection.memgraph_records import read_bundle
from apps.knowledge_graph.projection.memgraph_streaming import (
    stream_family_records,
    stream_projection_validation,
    validate_projection_count_limits,
)
from apps.knowledge_graph.projection.records import (
    CollectionGraphProjectionBundleV1,
    ProjectedDocumentMembershipV1,
    ProjectionCountsV1,
)
from apps.knowledge_graph.projection.serialization import projection_checksum
from apps.knowledge_graph.tests.test_projection_records import _bundle


class BundlePageDriver:
    def __init__(self, bundle) -> None:
        self.bundle = bundle
        self.calls = []

    def execute_read(self, cypher, parameters, *, timeout_seconds, max_records):
        self.calls.append((cypher, dict(parameters), timeout_seconds, max_records))
        if "CollectionGeneration" in cypher:
            return ({"record": asdict(self.bundle.generation)},)
        families = (
            ("ProjectedEntity", "entities", "entity_key"),
            ("AutomaticMembership", "automatic_memberships", "entity_key"),
            ("ProjectedDocument", "documents", "document_key"),
            ("ProjectedChunk", "chunks", "chunk_key"),
            ("ProjectedRelationSemantics", "relation_semantics", "semantics_key"),
            ("ProjectedRelation", "relations", "relation_key"),
            ("ProjectedEvidence", "evidence", "evidence_key"),
            ("ProjectedEntityMention", "entity_mentions", "mention_key"),
            ("ArtifactProvenance", "artifact_provenance", "scope_key"),
        )
        for label, field, identity in families:
            if f"(n:{label} " not in cypher:
                continue
            return tuple(
                {
                    "record": {
                        **asdict(row),
                        "opaque_key": getattr(row, identity),
                        "generation_key": self.bundle.generation.generation_key,
                    },
                    "cursor_id": index,
                }
                for index, row in enumerate(getattr(self.bundle, field))
            )
        raise AssertionError("unexpected streaming query")


class LargeDocumentPageDriver:
    def __init__(self, total: int, generation_key: str) -> None:
        self.total, self.generation_key, self.calls = total, generation_key, []

    def execute_read(self, cypher, parameters, *, timeout_seconds, max_records):
        del cypher, timeout_seconds
        self.calls.append((dict(parameters), max_records))
        start = 0
        if parameters["has_cursor"]:
            start = int(parameters["cursor_document_key"], 16) + 1
        stop = min(self.total, start + parameters["page_limit"])
        return tuple(
            {
                "record": {
                    "document_key": f"{index:064x}",
                    "generation_key": self.generation_key,
                    "opaque_key": f"{index:064x}",
                },
                "cursor_id": index,
            }
            for index in range(start, stop)
        )


class TopologyBundlePageDriver:
    def __init__(self, bundle) -> None:
        self.bundle, self.calls = bundle, []

    def execute_read(self, cypher, parameters, *, timeout_seconds, max_records):
        del timeout_seconds
        self.calls.append((cypher, dict(parameters), max_records))
        if "CollectionGeneration" in cypher:
            return ({"record": asdict(self.bundle.generation)},)
        families = (
            ("ProjectedEntity", "entities", "entity_key"),
            ("AutomaticMembership", "automatic_memberships", "entity_key"),
            ("ProjectedDocument", "documents", "document_key"),
            ("ProjectedChunk", "chunks", "chunk_key"),
            ("ProjectedRelationSemantics", "relation_semantics", "semantics_key"),
            ("ProjectedRelation", "relations", "relation_key"),
            ("ProjectedEvidence", "evidence", "evidence_key"),
            ("ProjectedEntityMention", "entity_mentions", "mention_key"),
            ("ArtifactProvenance", "artifact_provenance", "scope_key"),
        )
        for label, field, identity in families:
            if f"(n:{label} " not in cypher:
                continue
            records = getattr(self.bundle, field)
            start = parameters["cursor_id"] + 1 if parameters["has_cursor"] else 0
            stop = min(len(records), start + parameters["page_limit"])
            return tuple(
                {
                    "record": {
                        **asdict(records[index]),
                        "opaque_key": getattr(records[index], identity),
                        "generation_key": self.bundle.generation.generation_key,
                    },
                    "cursor_id": index,
                }
                for index in range(start, stop)
            )
        raise AssertionError("unexpected topology query")


def test_streamed_projection_checksum_matches_frozen_bundle_bytes() -> None:
    bundle = _bundle()
    driver = BundlePageDriver(bundle)

    result = stream_projection_validation(
        driver,
        generation_key=bundle.generation.generation_key,
        expected_counts=bundle.counts,
        timeout_seconds=1.0,
    )

    assert result.checksum == projection_checksum(bundle)
    assert result.counts == bundle.counts
    family_calls = [
        call for call in driver.calls if "CollectionGeneration" not in call[0]
    ]
    assert len(family_calls) == 9
    assert all(call[3] <= 5_000 for call in family_calls)
    assert all("LIMIT $page_limit" in call[0] for call in family_calls)


def test_exact_10000_document_scope_streams_without_5001_record_sentinel() -> None:
    generation_key = "1" * 64
    driver = LargeDocumentPageDriver(10_000, generation_key)

    count = sum(
        1
        for _row in stream_family_records(
            driver,
            generation_key=generation_key,
            label="ProjectedDocument",
            expected_count=10_000,
            timeout_seconds=1.0,
        )
    )

    assert count == 10_000
    assert len(driver.calls) == 11
    assert all(max_records <= 5_000 for _parameters, max_records in driver.calls)
    assert all(
        parameters["page_limit"] <= 5_000 for parameters, _limit in driver.calls
    )


def test_projection_validation_limits_admit_50k_entities_and_10k_documents() -> None:
    base = _bundle().counts
    supported = replace(
        base,
        entity_count=50_000,
        automatic_membership_count=50_000,
        document_count=10_000,
        artifact_provenance_count=10_001,
    )
    validate_projection_count_limits(supported)

    with pytest.raises(ValueError, match="supported validation limit"):
        validate_projection_count_limits(
            ProjectionCountsV1(
                50_001,
                50_000,
                10_000,
                base.chunk_count,
                base.relation_semantics_count,
                base.relation_count,
                base.evidence_count,
                base.entity_mention_count,
                10_001,
            )
        )


def test_shipping_bundle_paginates_exact_10k_authorized_scope() -> None:
    base = _bundle()
    document_keys = tuple(f"{index:064x}" for index in range(10_000))
    documents = tuple(
        ProjectedDocumentMembershipV1(key, base.generation.generation_key)
        for key in document_keys
    )
    collection_provenance = base.artifact_provenance[0]
    document_provenance = base.artifact_provenance[1]
    provenance = (
        collection_provenance,
        *(
            replace(
                document_provenance,
                scope_key=key,
                artifact_key=f"{index + 10_000:064x}",
            )
            for index, key in enumerate(document_keys)
        ),
    )
    bundle = CollectionGraphProjectionBundleV1(
        base.generation,
        base.entities,
        base.automatic_memberships,
        documents,
        base.chunks[:0],
        base.relation_semantics,
        base.relations,
        base.evidence[:0],
        base.entity_mentions[:0],
        provenance,
        replace(
            base.counts,
            document_count=10_000,
            chunk_count=0,
            evidence_count=0,
            entity_mention_count=0,
            artifact_provenance_count=10_001,
        ),
    )
    driver = TopologyBundlePageDriver(bundle)
    maxima = tuple(max(1, count) for count in asdict(bundle.counts).values())

    observed = read_bundle(
        driver,
        generation_key=bundle.generation.generation_key,
        maxima=maxima,
        timeout=1.0,
        reject_full_pages=True,
        topology_parameters={
            "seed_keys_csv": bundle.entities[0].entity_key,
            "max_depth": 2,
            "authorized_document_keys_csv": ",".join(document_keys),
            "collection_key": bundle.generation.collection_key,
        },
    )

    assert observed == bundle
    family_calls = [
        call for call in driver.calls if "CollectionGeneration" not in call[0]
    ]
    assert all(call[2] <= 5_000 for call in family_calls)
    assert all(call[1]["page_limit"] <= 5_000 for call in family_calls)
