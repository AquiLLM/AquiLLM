from __future__ import annotations

import uuid
from dataclasses import replace

import pytest

from apps.knowledge_graph.projection.django_projection_topology import (
    apply_relation_directions,
    expand_entity_mentions,
)
from apps.knowledge_graph.projection.identifiers import (
    HmacSha256ProjectionIdentifierCodec,
    ProjectionIdentifierDomain,
)
from apps.knowledge_graph.projection.projection_encoding import (
    encode_projection_snapshot,
)
from apps.knowledge_graph.projection.records import (
    CollectionGraphProjectionBundleV1,
    ProjectedEntityMentionEvidenceV1,
    ProjectedRelationSemanticsV1,
)
from apps.knowledge_graph.tests.test_django_projection_source import _snapshot
from apps.knowledge_graph.tests.test_projection_records import _bundle as _closed_bundle


def test_topology_input_records_are_closed_and_fail_closed() -> None:
    key = "a" * 64
    semantics = ProjectedRelationSemanticsV1(key, key, "works_at", "directed")
    mention = ProjectedEntityMentionEvidenceV1(
        key,
        key,
        key,
        key,
        key,
        2,
        0.75,
    )

    assert semantics.direction == "directed"
    assert mention.confidence == 0.75
    with pytest.raises(ValueError, match="direction"):
        replace(semantics, direction="sideways")
    with pytest.raises(ValueError, match="unit interval"):
        replace(mention, confidence=1.01)


@pytest.mark.parametrize("family", ("relations", "relation_semantics", "evidence"))
def test_projected_relation_types_are_canonical_retrieval_tokens(family) -> None:
    row = getattr(_closed_bundle(), family)[0]
    assert row.relation_type == "knows"
    with pytest.raises(ValueError, match="canonical token"):
        replace(row, relation_type="Works At")


def test_projection_encoding_binds_projection_semantics_and_isolated_mentions() -> None:
    projection_id = uuid.UUID("12345678-1234-5678-9234-567812345678")
    generation = uuid.UUID("22345678-1234-5678-9234-567812345678")
    snapshot = _snapshot(projection_id, generation)
    snapshot["relations"][0]["direction"] = "directed"
    snapshot["entity_mentions"] = (
        {
            "entity_id": 11,
            "mention_id": 51,
            "chunk_id": 101,
            "document_id": snapshot["documents"][0]["document_id"],
            "chunk_number": 2,
            "confidence": 0.625,
        },
    )
    codec = HmacSha256ProjectionIdentifierCodec(b"secret-a", key_version="key-v1")

    rows = encode_projection_snapshot(snapshot=snapshot, codec=codec)
    bundle = CollectionGraphProjectionBundleV1(**rows)

    assert bundle.generation.projection_key == codec.encode(
        ProjectionIdentifierDomain.COLLECTION,
        generation=generation,
        source=f"projection:{projection_id}",
    ).value
    assert tuple(row.direction for row in bundle.relation_semantics) == ("directed",)
    assert bundle.entity_mentions[0].entity_key == codec.encode(
        ProjectionIdentifierDomain.ENTITY, generation=generation, source=11
    ).value
    assert bundle.entity_mentions[0].chunk_key == bundle.chunks[0].chunk_key
    assert bundle.counts.relation_semantics_count == 1
    assert bundle.counts.entity_mention_count == 1


def test_relation_directions_require_exact_complete_artifact_semantics() -> None:
    relations = (
        {"id": 1, "relation_type": "works_at"},
        {"id": 2, "relation_type": "related_to"},
    )

    bound = apply_relation_directions(
        relations, {"works_at": "directed", "related_to": "undirected"}
    )

    assert tuple(row["direction"] for row in bound) == ("directed", "undirected")
    with pytest.raises(ValueError, match="complete"):
        apply_relation_directions(relations, {"works_at": "directed"})


def test_entity_mentions_include_isolated_representative_and_observation_chunks(
) -> None:
    document = uuid.UUID("12345678-1234-5678-9234-567812345678")
    rows = (
        {
            "entity_id": 11,
            "mention_id": 51,
            "chunk_id": 101,
            "document_id": document,
            "confidence": 0.625,
            "metadata": {"observations": [{"chunk_id": 102}]},
        },
    )
    chunks = {
        101: (document, 2),
        102: (document, 3),
    }

    expanded = expand_entity_mentions(rows, chunks)

    assert expanded == (
        {
            "entity_id": 11,
            "mention_id": 51,
            "chunk_id": 101,
            "document_id": document,
            "chunk_number": 2,
            "confidence": 0.625,
        },
        {
            "entity_id": 11,
            "mention_id": 51,
            "chunk_id": 102,
            "document_id": document,
            "chunk_number": 3,
            "confidence": 0.625,
        },
    )
