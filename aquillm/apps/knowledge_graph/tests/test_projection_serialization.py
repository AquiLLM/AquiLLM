from dataclasses import astuple, replace
from datetime import UTC, datetime
from hashlib import sha256

import pytest

from apps.knowledge_graph.projection.records import (
    PrivateProjectionChunkReferenceV1,
    ProjectedEntityV1,
    ProjectionCountsV1,
    ProjectionFailureCode,
    ProjectionFailureStateV1,
    ProjectionGenerationManifestV1,
    ProjectionLeaseV1,
    ProjectionLifecycleState,
)
from apps.knowledge_graph.projection.serialization import (
    canonical_projection_bytes,
    private_chunk_mapping_checksum,
    projection_checksum,
)
from apps.knowledge_graph.tests.test_projection_records import _bundle

GENERATION, COLLECTION, ARTIFACT, ENTITY_A, ENTITY_B = (
    character * 64 for character in "01234"
)
CLUSTER = "a" * 64
DOCUMENT_UUID = "12345678-1234-5678-9234-567812345678"
ZERO_COUNTS = ProjectionCountsV1(0, 0, 0, 0, 0, 0, 0, 0, 0)


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


def _lease(owner="worker-a") -> ProjectionLeaseV1:
    return ProjectionLeaseV1(
        DOCUMENT_UUID, owner, datetime(2026, 8, 20, 12, tzinfo=UTC), 2
    )


def _private(key="5", pk=19, uuid=DOCUMENT_UUID, chunk=2):
    return PrivateProjectionChunkReferenceV1(key * 64, pk, uuid, chunk)


def test_private_mapping_literal_checksum_and_order_are_separate() -> None:
    first = _private()
    second = _private("6", 23, "22345678-1234-5678-9234-567812345678", 0)
    assert private_chunk_mapping_checksum((first, second)) == (
        "7e80265013215298d1c86019f83c556c2a92d3825446edd046a65e5346c208cf"
    )
    with pytest.raises(ValueError, match="sorted"):
        private_chunk_mapping_checksum((second, first))
    with pytest.raises(ValueError, match="unique"):
        private_chunk_mapping_checksum((first, first))

    conflicts = (
        replace(first, integer_chunk_pk=20),
        replace(second, integer_chunk_pk=19),
        replace(second, document_uuid=DOCUMENT_UUID, chunk_number=2),
    )
    for conflict in conflicts:
        rows = tuple(sorted((first, conflict), key=astuple))
        with pytest.raises(ValueError, match="unique"):
            private_chunk_mapping_checksum(rows)


def test_literal_manifest_lifecycle_lease_failure_and_full_bundle_vectors() -> None:
    manifest = ProjectionGenerationManifestV1(
        generation_key=GENERATION,
        schema_version="schema-v1",
        projection_version="projection-v1",
        identifier_key_version="key-v1",
        graph_checksum="a" * 64,
        snapshot_checksum="b" * 64,
        private_mapping_checksum="c" * 64,
        counts=ZERO_COUNTS,
        state=ProjectionLifecycleState.READY,
    )
    lease = _lease()
    failure = ProjectionFailureStateV1(
        state=ProjectionLifecycleState.FAILED,
        failure_code=ProjectionFailureCode.CHECKSUM_MISMATCH,
        attempt_count=2,
    )
    expected_manifest = (
        b'{"counts":{"artifact_provenance_count":0,"automatic_membership_count":0,"chu'
        b'nk_count":0,"document_count":0,"entity_count":0,"entity_mention_count":0,"e'
        b'vidence_count":0,"relation_count":0,"relation_semantics_count":0},"generatio'
        b'n_key":"<GEN>","graph_checksum":"<GRAPH>","identifier_k'
        b'ey_version":"key-v1","private_mapping_checksum":"<PRIVATE>","projection_vers'
        b'ion":"projection-v1","schema_version":"schema-v1","snapshot_checksum":"<SNAP'
        b'>","state":"ready"}'
    )
    replacements = (("GEN", "0"), ("GRAPH", "a"), ("SNAP", "b"), ("PRIVATE", "c"))
    for marker, character in replacements:
        expected_manifest = expected_manifest.replace(
            f"<{marker}>".encode(), (character * 64).encode()
        )
    expected_bundle = (
        b'{"artifact_provenance":[{"artifact_key":"<ART>","assembly_config_checksum":"'
        b'<F>","assembly_version":"assembly-v1","build_generation":3,"build_key":"<BUI'
        b'LD>","collection_key":"<COL>","embedding_model_signature":"embed-v1","evalua'
        b'tion_only":false,"extractor_version":"extractor-v1","filter_policy_checksum"'
        b':"<F>","filter_policy_version":"filter-v1","ontology_checksum":"<F>","ontolo'
        b'gy_version":"ontology-v1","orchestration_version":4,"rebuild_request_key":nu'
        b'll,"resolution_config_checksum":"<F>","resolver_version":"resolver-v1","scop'
        b'e_key":"<COL>","scope_type":"collection","source_hash":"<F>"},{"artifact_key'
        b'":"<F>","assembly_config_checksum":"<REBUILD>","assembly_version":"not-appli'
        b'cable","build_generation":3,"build_key":"<BUILD>","collection_key":"<COL>","'
        b'embedding_model_signature":"","evaluation_only":false,"extractor_version":"e'
        b'xtractor-v1","filter_policy_checksum":"<F>","filter_policy_version":"filter-'
        b'v1","ontology_checksum":"<F>","ontology_version":"ontology-v1","orchestratio'
        b'n_version":4,"rebuild_request_key":null,"resolution_config_checksum":"<F>","'
        b'resolver_version":"resolver-v1","scope_key":"<DOC>","scope_type":"document",'
        b'"source_hash":"<F>"}],"automatic_memberships":[{"automatic_membership_key":nu'
        b'll,"decision_checksum":"<F>","entity_key":"<EA>","resolution_config_checksum'
        b'":"<F>","resolver_version":"resolver-v1"},{"automatic_membership_key":"<AUTO'
        b'>","decision_checksum":"<F>","entity_key":"<EB>","resolution_config_checksum'
        b'":"<F>","resolver_version":"resolver-v1"}],"chunks":[{"chunk_key":"<CHUNK>",'
        b'"chunk_number":2,"document_key":"<DOC>"}],"counts":{"artifact_provenance_cou'
        b'nt":2,"automatic_membership_count":2,"chunk_count":1,"document_count":1,"ent'
        b'ity_count":2,"entity_mention_count":1,"evidence_count":1,"relation_count":1'
        b',"relation_semantics_count":1},"documents":[{"document_'
        b'key":"<DOC>","generation_key":"<GEN>"}],"entities":[{"artifact_key":"<ART>",'
        b'"cluster_key":"<AUTO>","collection_key":"<COL>","entity_key":"<EA>","generat'
        b'ion_key":"<GEN>","ontology_type":"person","retrieval_utility":"0x1.000000000'
        b'0000p-1"},{"artifact_key":"<ART>","cluster_key":"<AUTO>","collection_key":"<'
        b'COL>","entity_key":"<EB>","generation_key":"<GEN>","ontology_type":"person",'
        b'"retrieval_utility":"0x1.0000000000000p-2"}],"entity_mentions":[{"chunk_key"'
        b':"<CHUNK>","chunk_number":2,"confidence":"0x1.0000000000000p-1","document_'
        b'key":"<DOC>","entity_key":"<EA>","mention_key":"<MENT>","provenance_key":"'
        b'<REBUILD>"}],"evidence":[{"artifact_key":"<F'
        b'>","assembly_config_checksum":"<F>","chunk_key":"<CHUNK>","chunk_number":2,"'
        b'confidence":"0x1.8000000000000p-1","document_key":"<DOC>","evidence_key":"<E'
        b'VID>","head_mapping_key":"<EA>","head_mention_key":"<MENT>","ontology_checks'
        b'um":"<F>","orientation":"head_to_tail","provenance_key":"<PROV>","relation_k'
        b'ey":"<REL>","relation_mention_key":"<MENT>","relation_type":"knows","semanti'
        b'c_signature":"<F>","source_document_key":"<DOC>","tail_mapping_key":"<EB>","'
        b'tail_mention_key":"<EVID>"}],"generation":{"artifact_key":"<ART>","collectio'
        b'n_key":"<COL>","generation_key":"<GEN>","identifier_key_version":"key-v1","m'
        b'embership_checksum":"<F>","membership_epoch":7,"projection_key":"<SCOPE>"'
        b',"projection_version":"projection-v1","schema_version":"memgraph-schema-v1"'
        b'},"relation_semantics":[{"artifact_key":"<ART>","direction":"directed","rel'
        b'ation_type":"knows","semantics_key":"<SCOPE>"}],"relations":[{"artifact_key":'
        b'"<ART>","relation_key":"<REL>","relation_type":"knows","source_entity_key":"'
        b'<EA>","target_entity_key":"<EB>"}]}'
    )
    markers = (
        "GEN COL ART EA EB DOC CHUNK REL EVID MENT PROV AUTO SCOPE BUILD REBUILD F"
    ).split()
    for marker, character in zip(markers, "0123456789abcdef", strict=True):
        expected_bundle = expected_bundle.replace(
            f"<{marker}>".encode(), (character * 64).encode()
        )
    assert canonical_projection_bytes(manifest) == expected_manifest
    assert canonical_projection_bytes(lease) == (
        b'{"attempt_count":2,"expires_at":"2026-08-20T12:00:00Z","owner":"worker-a","p'
        b'rojection_id":"12345678-1234-5678-9234-567812345678"}'
    )
    assert canonical_projection_bytes(failure) == (
        b'{"attempt_count":2,"failure_code":"checksum_mismatch","state":"failed"}'
    )
    assert canonical_projection_bytes(_bundle()) == expected_bundle
    assert tuple(map(projection_checksum, (manifest, lease, failure, _bundle()))) == (
        "b48738ee41c9594aec2e691194c5e3078dad80235a757def897f3b40299a0a52",
        "dbe87cb726e6db021d018061ddf1eee20f192c7fcf13b72242aabc06f2a8229d",
        "63a322de2d531a1ed7ab3dd19baa7cb907fa1eebf969fc93a19d8ce7b81bc7db",
        "e4732ed48ebcca6a9a13ae4719936ba044828bcac029a9a5924acc5abe9d5788",
    )
    assert " ".join(state.value for state in ProjectionLifecycleState) == (
        "pending building ready failed superseded"
    )
    assert " ".join(code.value for code in ProjectionFailureCode) == (
        "source_changed lease_lost graph_unavailable write_failed validation_failed "
        "checksum_mismatch timeout internal_error"
    )
    for record in (lease, failure):
        for attempt in (True, 32768):
            with pytest.raises((TypeError, ValueError)):
                replace(record, attempt_count=attempt)


def test_private_integer_pk_cannot_enter_provider_neutral_bytes() -> None:
    private = _private()
    with pytest.raises(TypeError, match="provider-neutral"):
        canonical_projection_bytes(private)
    with pytest.raises(TypeError):
        canonical_projection_bytes({"integer_chunk_pk": 19})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("orientation", "forward"),
        ("orientation", type("_StrSubclass", (str,), {})("head_to_tail")),
        ("relation_type", ""),
        ("relation_type", type("_StrSubclass", (str,), {})("knows")),
    ],
)
def test_evidence_orientation_and_relation_type_are_strict(field, value) -> None:
    with pytest.raises((TypeError, ValueError), match=field):
        replace(_bundle().evidence[0], **{field: value})


@pytest.mark.parametrize(
    ("factory", "error"),
    [
        (lambda: replace(_bundle().counts, entity_count=True), TypeError),
        (lambda: replace(_bundle().counts, entity_count=-1), ValueError),
        (lambda: replace(_bundle().chunks[0], chunk_number=-1), ValueError),
        (lambda: replace(_bundle().entities[0], entity_key="A" * 64), ValueError),
        (lambda: replace(_bundle().entities[0], cluster_key="cluster-a"), ValueError),
        (lambda: replace(_bundle().generation, schema_version=""), ValueError),
        (lambda: replace(_bundle().entities[0], entity_key=None), TypeError),
        (
            lambda: replace(_bundle().entities[0], ontology_type="per\x00son"),
            ValueError,
        ),
        (lambda: replace(_bundle().relations[0], relation_type="know\ns"), ValueError),
        (lambda: _lease("worker\x7f"), ValueError),
        (lambda: _private(uuid="AAAAAAAA-1234-5678-9234-567812345678"), ValueError),
        (
            lambda: replace(
                _bundle().artifact_provenance[0],
                scope_type=type("_StringSubclass", (str,), {})("collection"),
            ),
            TypeError,
        ),
    ],
)
def test_records_reject_exact_type_bounds_digest_token_and_uuid(factory, error):
    with pytest.raises(error):
        factory()
    assert (
        replace(_bundle().entities[0], ontology_type="pérson").ontology_type == "pérson"
    )


@pytest.mark.parametrize(("integer_pk", "error"), [(True, TypeError), (0, ValueError)])
def test_private_reference_integer_pk_is_exact_and_positive(integer_pk, error) -> None:
    with pytest.raises(error):
        _private(pk=integer_pk)


@pytest.mark.parametrize(
    ("value", "error"),
    [
        (0.0, None),
        (1.0, None),
        (-0.01, ValueError),
        (1.01, ValueError),
        (float("nan"), ValueError),
        (float("inf"), ValueError),
        (float("-inf"), ValueError),
        (True, TypeError),
        (0, TypeError),
    ],
)
def test_projection_scores_are_exact_finite_unit_interval_floats(value, error) -> None:
    records = (
        (_entity(ENTITY_A, 0.5), "retrieval_utility"),
        (_bundle().evidence[0], "confidence"),
    )
    for record, field in records:
        if error is None:
            assert getattr(replace(record, **{field: value}), field) == value
        else:
            with pytest.raises(error, match=field):
                replace(record, **{field: value})


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
