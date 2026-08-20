from dataclasses import replace
from hashlib import sha256

import pytest

from apps.knowledge_graph.projection.records import (
    AutomaticCanonicalMembershipV1,
    CollectionGraphProjectionBundleV1,
    PrivateProjectionChunkReferenceV1,
    ProjectedEntityV1,
    ProjectionCountsV1,
    ProjectionGenerationMarkerV1,
)
from apps.knowledge_graph.projection.serialization import (
    canonical_projection_bytes,
    private_chunk_mapping_checksum,
    projection_checksum,
)

GENERATION = "0" * 64
COLLECTION = "1" * 64
ARTIFACT = "2" * 64
ENTITY_A = "3" * 64
ENTITY_B = "4" * 64
CLUSTER = "a" * 64
DIGEST = "f" * 64
DOCUMENT_UUID = "12345678-1234-5678-9234-567812345678"


def _entity(key: str, utility: float) -> ProjectedEntityV1:
    return ProjectedEntityV1(
        entity_key=key,
        generation_key=GENERATION,
        artifact_key=ARTIFACT,
        collection_key=COLLECTION,
        ontology_type="person",
        cluster_key=CLUSTER,
        retrieval_utility=utility,
    )


def _empty_bundle() -> CollectionGraphProjectionBundleV1:
    return CollectionGraphProjectionBundleV1(
        generation=ProjectionGenerationMarkerV1(
            generation_key=GENERATION,
            collection_key=COLLECTION,
            artifact_key=ARTIFACT,
            schema_version="schema-v1",
            projection_version="projection-v1",
            identifier_key_version="key-v1",
            membership_epoch=0,
            membership_checksum=DIGEST,
        ),
        entities=(),
        automatic_memberships=(),
        documents=(),
        chunks=(),
        relations=(),
        evidence=(),
        artifact_provenance=(),
        counts=ProjectionCountsV1(
            entity_count=0,
            automatic_membership_count=0,
            document_count=0,
            chunk_count=0,
            relation_count=0,
            evidence_count=0,
            artifact_provenance_count=0,
        ),
    )


def test_literal_entity_array_uses_sorted_keys_utf8_and_float_hex() -> None:
    records = (_entity(ENTITY_A, 0.5), _entity(ENTITY_B, 0.25))
    expected = (
        '[{"artifact_key":"'
        + ARTIFACT
        + '","cluster_key":"'
        + CLUSTER
        + '","collection_key":"'
        + COLLECTION
        + '","entity_key":"'
        + ENTITY_A
        + '","generation_key":"'
        + GENERATION
        + '","ontology_type":"person","retrieval_utility":"0x1.0000000000000p-1"},'
        + '{"artifact_key":"'
        + ARTIFACT
        + '","cluster_key":"'
        + CLUSTER
        + '","collection_key":"'
        + COLLECTION
        + '","entity_key":"'
        + ENTITY_B
        + '","generation_key":"'
        + GENERATION
        + '","ontology_type":"person","retrieval_utility":"0x1.0000000000000p-2"}]'
    ).encode()
    assert canonical_projection_bytes(records) == expected
    assert projection_checksum(records) == (
        "e4969a0ca5bd06de66b4b703ded731d20abbca108d6b45f6f512e25ea62e1bb6"
    )


def test_literal_null_membership_is_explicit() -> None:
    record = AutomaticCanonicalMembershipV1(
        entity_key=ENTITY_A,
        automatic_membership_key=None,
        decision_checksum=DIGEST,
        resolver_version="résolver-v1",
        resolution_config_checksum=DIGEST,
    )
    expected = (
        '{"automatic_membership_key":null,"decision_checksum":"'
        + DIGEST
        + '","entity_key":"'
        + ENTITY_A
        + '","resolution_config_checksum":"'
        + DIGEST
        + '","resolver_version":"résolver-v1"}'
    ).encode()
    assert canonical_projection_bytes(record) == expected


def test_literal_empty_bundle_and_checksum_pin_versions_counts_and_arrays() -> None:
    expected = (
        '{"artifact_provenance":[],"automatic_memberships":[],"chunks":[],'
        '"counts":{"artifact_provenance_count":0,"automatic_membership_count":0,'
        '"chunk_count":0,"document_count":0,"entity_count":0,"evidence_count":0,'
        '"relation_count":0},"documents":[],"entities":[],"evidence":[],'
        '"generation":{"artifact_key":"'
        + ARTIFACT
        + '","collection_key":"'
        + COLLECTION
        + '","generation_key":"'
        + GENERATION
        + '","identifier_key_version":"key-v1","membership_checksum":"'
        + DIGEST
        + '","membership_epoch":0,"projection_version":"projection-v1",'
        '"schema_version":"schema-v1"},"relations":[]}'
    ).encode()
    assert canonical_projection_bytes(_empty_bundle()) == expected
    assert projection_checksum(_empty_bundle()) == (
        "4b5121f7b5c7e9a667aa03eb84966212554fa6892e6acc6b7dddfb84af1ea325"
    )
    assert sha256(expected + b"x").hexdigest() != projection_checksum(_empty_bundle())


def test_private_mapping_literal_checksum_and_order_are_separate() -> None:
    first = PrivateProjectionChunkReferenceV1(
        projection_chunk_key="5" * 64,
        integer_chunk_pk=19,
        document_uuid=DOCUMENT_UUID,
        chunk_number=2,
    )
    second = PrivateProjectionChunkReferenceV1(
        projection_chunk_key="6" * 64,
        integer_chunk_pk=23,
        document_uuid="22345678-1234-5678-9234-567812345678",
        chunk_number=0,
    )
    assert private_chunk_mapping_checksum((first, second)) == (
        "7e80265013215298d1c86019f83c556c2a92d3825446edd046a65e5346c208cf"
    )
    with pytest.raises(ValueError, match="sorted"):
        private_chunk_mapping_checksum((second, first))
    with pytest.raises(ValueError, match="unique"):
        private_chunk_mapping_checksum((first, first))


def test_private_integer_pk_cannot_enter_provider_neutral_bytes() -> None:
    private = PrivateProjectionChunkReferenceV1(
        projection_chunk_key="5" * 64,
        integer_chunk_pk=19,
        document_uuid=DOCUMENT_UUID,
        chunk_number=2,
    )
    with pytest.raises(TypeError, match="provider-neutral"):
        canonical_projection_bytes(private)
    with pytest.raises(TypeError):
        canonical_projection_bytes({"integer_chunk_pk": 19})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_numeric_values_are_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        _entity(ENTITY_A, value)


def test_float_byte_and_record_order_mutations_change_or_fail_contract() -> None:
    records = (_entity(ENTITY_A, 0.5), _entity(ENTITY_B, 0.25))
    changed = (replace(records[0], retrieval_utility=0.5000000000000001), records[1])
    assert canonical_projection_bytes(changed) != canonical_projection_bytes(records)
    with pytest.raises(ValueError, match="sorted"):
        canonical_projection_bytes(tuple(reversed(records)))
    mutated = bytearray(canonical_projection_bytes(records))
    mutated[-2] ^= 1
    assert sha256(mutated).hexdigest() != projection_checksum(records)


@pytest.mark.parametrize(
    "value",
    [True, 7, "opaque", [ENTITY_A], (ENTITY_A,), object()],
)
def test_serializer_rejects_unknown_types_and_arbitrary_stringification(value) -> None:
    with pytest.raises(TypeError, match="supported"):
        canonical_projection_bytes(value)


def test_duplicate_and_unsorted_record_arrays_are_rejected() -> None:
    first = _entity(ENTITY_A, 0.5)
    second = _entity(ENTITY_B, 0.25)
    with pytest.raises(ValueError, match="unique"):
        canonical_projection_bytes((first, first))
    with pytest.raises(ValueError, match="sorted"):
        canonical_projection_bytes((second, first))
