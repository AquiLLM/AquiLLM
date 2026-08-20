from __future__ import annotations

import uuid

import pytest

from apps.knowledge_graph.projection.django_projection_source import (
    DjangoProjectionRowSource,
)
from apps.knowledge_graph.projection.identifiers import (
    HmacSha256ProjectionIdentifierCodec,
    ProjectionIdentifierDomain,
)
from apps.knowledge_graph.projection.memberships import (
    membership_decision_checksum,
    null_membership_decision_checksum,
)
from apps.knowledge_graph.projection.records import (
    AutomaticCanonicalMembershipV1,
    CollectionGraphProjectionBundleV1,
    PrivateProjectionChunkReferenceV1,
)

H = "a" * 64


class _Loader:
    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.calls = []

    def load(self, *, projection_id, batch_size):
        self.calls.append((projection_id, batch_size))
        return self.snapshot


def _artifact(artifact_id, scope_type, scope_id, *, embedding):
    return {
        "id": artifact_id,
        "scope_type": scope_type,
        "scope_id": str(scope_id),
        "collection_id": 7,
        "rebuild_request_id": None,
        "evaluation_only": False,
        "build_key": H,
        "build_generation": 1,
        "orchestration_version": 1,
        "source_hash": H,
        "ontology_version": "ontology-v1",
        "ontology_checksum": H,
        "extractor_version": "extractor-v1",
        "resolver_version": "resolver-v1",
        "resolution_config_checksum": H,
        "filter_policy_version": "filter-v1",
        "filter_policy_checksum": H,
        "embedding_model_signature": embedding,
        "assembly_version": "assembly-v1" if scope_type == "collection" else "na-v1",
        "assembly_config_checksum": H,
    }


def _snapshot(projection_id, generation):
    document = uuid.UUID("12345678-1234-5678-9234-567812345678")
    snapshot = {
        "projection": {
            "id": projection_id,
            "generation_key": generation,
            "collection_id": 7,
            "artifact_id": 9,
            "state": "building",
            "schema_version": "schema-v1",
            "projection_version": "projection-v1",
            "identifier_key_version": "key-v1",
            "membership_epoch": 4,
            "membership_checksum": "0" * 64,
        },
        "artifacts": (
            _artifact(9, "collection", 7, embedding="embed-v1"),
            _artifact(10, "document", document, embedding=""),
        ),
        "entities": (
            {
                "id": 11,
                "artifact_id": 9,
                "collection_id": 7,
                "entity_type": "person",
                "cluster_key": "b" * 64,
                "retrieval_utility": 0.5,
            },
            {
                "id": 12,
                "artifact_id": 9,
                "collection_id": 7,
                "entity_type": "person",
                "cluster_key": "c" * 64,
                "retrieval_utility": 0.25,
            },
        ),
        "memberships": (
            {
                "entity_id": 11,
                "canonical_entity_id": 91,
                "outcome": "automatic",
                "status": "active",
                "canonical_status": "active",
                "decision_checksum": "d" * 64,
            },
            {
                "entity_id": 12,
                "canonical_entity_id": 92,
                "outcome": "candidate",
                "status": "suppressed",
                "canonical_status": "active",
            },
        ),
        "documents": ({"document_id": document, "artifact_id": 10},),
        "chunks": ({"id": 101, "document_id": document, "chunk_number": 2},),
        "relations": (
            {
                "id": 21,
                "artifact_id": 9,
                "source_id": 11,
                "relation_type": "knows",
                "target_id": 12,
            },
        ),
        "evidence": (
            {
                "id": 31,
                "relation_id": 21,
                "relation_mention_id": 41,
                "chunk_id": 101,
                "document_id": document,
                "chunk_number": 2,
                "confidence": 0.75,
                "artifact_id": 10,
                "head_mention_id": 51,
                "tail_mention_id": 52,
                "head_mapping_id": 61,
                "tail_mapping_id": 62,
                "orientation": "head_to_tail",
                "relation_type": "knows",
                "ontology_checksum": H,
                "assembly_config_checksum": H,
            },
        ),
    }
    codec = HmacSha256ProjectionIdentifierCodec(b"secret-a", key_version="key-v1")
    keys = {
        entity_id: codec.encode(
            ProjectionIdentifierDomain.ENTITY,
            generation="membership-registry-v1",
            source=entity_id,
        ).value
        for entity_id in (11, 12)
    }
    audit = (
        AutomaticCanonicalMembershipV1(
            keys[11],
            codec.encode(
                ProjectionIdentifierDomain.AUTOMATIC_CANONICAL_IDENTITY,
                source=91,
            ).value,
            "d" * 64,
            "resolver-v1",
            H,
        ),
        AutomaticCanonicalMembershipV1(
            keys[12],
            None,
            null_membership_decision_checksum(keys[12], "resolver-v1", H),
            "resolver-v1",
            H,
        ),
    )
    snapshot["projection"]["membership_checksum"] = membership_decision_checksum(
        tuple(sorted(audit, key=lambda row: row.entity_key))
    )
    return snapshot


def test_source_encodes_all_authoritative_families_and_exact_private_rows() -> None:
    projection_id = uuid.uuid4()
    generation = uuid.uuid4()
    loader = _Loader(_snapshot(projection_id, generation))
    source = DjangoProjectionRowSource(
        using="graph_reader",
        loader=loader,
        identifier_key=b"secret-a",
        identifier_key_version="key-v1",
    )

    rows = source.load_projection_rows(projection_id=projection_id, batch_size=37)
    private = source.load_private_chunk_rows(projection_id=projection_id, batch_size=37)

    assert loader.calls == [(projection_id, 37), (projection_id, 37)]
    assert (
        type(CollectionGraphProjectionBundleV1(**rows))
        is CollectionGraphProjectionBundleV1
    )
    codec = HmacSha256ProjectionIdentifierCodec(b"secret-a", key_version="key-v1")
    assert (
        rows["generation"].generation_key
        == codec.encode(
            ProjectionIdentifierDomain.COLLECTION,
            generation=generation,
            source=generation,
        ).value
    )
    assert {name: len(rows[name]) for name in rows if type(rows[name]) is tuple} == {
        "entities": 2,
        "automatic_memberships": 2,
        "documents": 1,
        "chunks": 1,
        "relations": 1,
        "evidence": 1,
        "artifact_provenance": 2,
    }
    assert {
        row.automatic_membership_key is None for row in rows["automatic_memberships"]
    } == {
        False,
        True,
    }
    assert {row.decision_checksum for row in rows["automatic_memberships"]} == {
        rows["generation"].membership_checksum
    }
    assert private == (
        PrivateProjectionChunkReferenceV1(
            rows["chunks"][0].chunk_key,
            101,
            str(uuid.UUID("12345678-1234-5678-9234-567812345678")),
            2,
        ),
    )
    assert "label" not in repr(rows).lower()
    assert "content" not in repr(rows).lower()


def test_source_rejects_membership_audit_drift() -> None:
    projection_id = uuid.uuid4()
    snapshot = _snapshot(projection_id, uuid.uuid4())
    snapshot["memberships"][0]["decision_checksum"] = "e" * 64
    source = DjangoProjectionRowSource(
        using="graph_reader",
        loader=_Loader(snapshot),
        identifier_key=b"secret-a",
        identifier_key_version="key-v1",
    )

    with pytest.raises(ValueError, match="membership audit"):
        source.load_projection_rows(projection_id=projection_id, batch_size=37)
