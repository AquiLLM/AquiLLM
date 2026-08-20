from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime

import pytest

from apps.knowledge_graph.projection.records import (
    AutomaticCanonicalMembershipV1,
    CollectionGraphProjectionBundleV1,
    PrivateProjectionChunkReferenceV1,
    ProjectedArtifactProvenanceV1,
    ProjectedChunkMembershipV1,
    ProjectedDocumentMembershipV1,
    ProjectedEntityV1,
    ProjectedPhysicalRelationV1,
    ProjectedRelationEvidenceV1,
    ProjectionCountsV1,
    ProjectionFailureCode,
    ProjectionFailureStateV1,
    ProjectionGenerationManifestV1,
    ProjectionGenerationMarkerV1,
    ProjectionLeaseV1,
    ProjectionLifecycleState,
)

_NAMES = (
    "generation collection artifact entity_a entity_b document chunk relation "
    "evidence mention provenance automatic scope build rebuild"
).split()
K = dict(zip(_NAMES, (character * 64 for character in "0123456789abcde"), strict=True))
DIGEST = "f" * 64
DOCUMENT_UUID = "12345678-1234-5678-9234-567812345678"


def _bundle() -> CollectionGraphProjectionBundleV1:
    marker = ProjectionGenerationMarkerV1(
        generation_key=K["generation"],
        collection_key=K["collection"],
        artifact_key=K["artifact"],
        schema_version="memgraph-schema-v1",
        projection_version="projection-v1",
        identifier_key_version="key-v1",
        membership_epoch=7,
        membership_checksum=DIGEST,
    )
    entities = tuple(
        ProjectedEntityV1(
            entity_key=K[name],
            generation_key=K["generation"],
            artifact_key=K["artifact"],
            collection_key=K["collection"],
            ontology_type="person",
            cluster_key=K["automatic"],
            retrieval_utility=utility,
        )
        for name, utility in (("entity_a", 0.5), ("entity_b", 0.25))
    )
    automatic = (
        AutomaticCanonicalMembershipV1(
            entity_key=K["entity_a"],
            automatic_membership_key=None,
            decision_checksum=DIGEST,
            resolver_version="resolver-v1",
            resolution_config_checksum=DIGEST,
        ),
    )
    documents = (
        ProjectedDocumentMembershipV1(
            document_key=K["document"],
            generation_key=K["generation"],
        ),
    )
    chunks = (
        ProjectedChunkMembershipV1(
            chunk_key=K["chunk"],
            document_key=K["document"],
            chunk_number=2,
        ),
    )
    relations = (
        ProjectedPhysicalRelationV1(
            relation_key=K["relation"],
            artifact_key=K["artifact"],
            source_entity_key=K["entity_a"],
            relation_type="knows",
            target_entity_key=K["entity_b"],
        ),
    )
    evidence = (
        ProjectedRelationEvidenceV1(
            evidence_key=K["evidence"],
            relation_key=K["relation"],
            relation_mention_key=K["mention"],
            chunk_key=K["chunk"],
            document_key=K["document"],
            chunk_number=2,
            confidence=0.75,
            source_document_key=K["document"],
            head_mention_key=K["mention"],
            tail_mention_key=K["evidence"],
            head_mapping_key=K["entity_a"],
            tail_mapping_key=K["entity_b"],
            orientation="head_to_tail",
            ontology_checksum=DIGEST,
            assembly_config_checksum=DIGEST,
            provenance_key=K["provenance"],
            semantic_signature=DIGEST,
        ),
    )
    provenance = (
        ProjectedArtifactProvenanceV1(
            artifact_key=K["artifact"],
            scope_type="collection",
            scope_key=K["scope"],
            collection_key=K["collection"],
            rebuild_request_key=None,
            evaluation_only=False,
            build_key=K["build"],
            build_generation=3,
            orchestration_version=4,
            source_hash=DIGEST,
            ontology_version="ontology-v1",
            ontology_checksum=DIGEST,
            extractor_version="extractor-v1",
            resolver_version="resolver-v1",
            resolution_config_checksum=DIGEST,
            filter_policy_version="filter-v1",
            filter_policy_checksum=DIGEST,
            embedding_model_signature="embed-v1",
            assembly_version="assembly-v1",
            assembly_config_checksum=DIGEST,
        ),
    )
    counts = ProjectionCountsV1(
        entity_count=2,
        automatic_membership_count=1,
        document_count=1,
        chunk_count=1,
        relation_count=1,
        evidence_count=1,
        artifact_provenance_count=1,
    )
    return CollectionGraphProjectionBundleV1(
        generation=marker,
        entities=entities,
        automatic_memberships=automatic,
        documents=documents,
        chunks=chunks,
        relations=relations,
        evidence=evidence,
        artifact_provenance=provenance,
        counts=counts,
    )


def test_records_are_immutable_slotted_and_bundle_is_closed() -> None:
    bundle = _bundle()
    assert bundle.counts.evidence_count == 1
    assert bundle.automatic_memberships[0].automatic_membership_key is None
    assert not hasattr(bundle.entities[0], "__dict__")
    with pytest.raises(FrozenInstanceError):
        bundle.counts.entity_count = 9  # type: ignore[misc]


def test_manifest_keeps_checksum_roles_and_versions_distinct() -> None:
    manifest = ProjectionGenerationManifestV1(
        generation_key=K["generation"],
        schema_version="memgraph-schema-v1",
        projection_version="projection-v1",
        identifier_key_version="key-v1",
        graph_checksum="a" * 64,
        snapshot_checksum="b" * 64,
        private_mapping_checksum="c" * 64,
        counts=_bundle().counts,
        state=ProjectionLifecycleState.READY,
    )
    assert (
        len(
            {
                manifest.graph_checksum,
                manifest.snapshot_checksum,
                manifest.private_mapping_checksum,
            }
        )
        == 3
    )


def test_lifecycle_lease_and_failure_contracts_are_closed_and_bounded() -> None:
    assert {state.value for state in ProjectionLifecycleState} == {
        "pending",
        "building",
        "ready",
        "failed",
        "superseded",
    }
    assert {code.value for code in ProjectionFailureCode} == {
        "source_changed",
        "lease_lost",
        "graph_unavailable",
        "write_failed",
        "validation_failed",
        "checksum_mismatch",
        "timeout",
        "internal_error",
    }
    lease = ProjectionLeaseV1(
        projection_id=DOCUMENT_UUID,
        owner="worker-a",
        expires_at=datetime(2026, 8, 20, 12, tzinfo=UTC),
        attempt_count=2,
    )
    failure = ProjectionFailureStateV1(
        state=ProjectionLifecycleState.FAILED,
        failure_code=ProjectionFailureCode.CHECKSUM_MISMATCH,
        attempt_count=2,
    )
    assert failure.failure_code.value == "checksum_mismatch"
    with pytest.raises((TypeError, ValueError)):
        replace(lease, attempt_count=True)


@pytest.mark.parametrize("integer_pk", [0, True])
def test_private_mapping_record_validates_integer_pk(integer_pk) -> None:
    with pytest.raises((TypeError, ValueError)):
        PrivateProjectionChunkReferenceV1(K["chunk"], integer_pk, DOCUMENT_UUID, 2)


def test_private_mapping_record_is_an_explicit_postgres_only_tuple() -> None:
    row = PrivateProjectionChunkReferenceV1(K["chunk"], 19, DOCUMENT_UUID, 2)
    assert (
        row.projection_chunk_key,
        row.integer_chunk_pk,
        row.document_uuid,
        row.chunk_number,
    ) == (K["chunk"], 19, DOCUMENT_UUID, 2)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("entities", lambda b: tuple(reversed(b.entities)), "sorted"),
        ("entities", lambda b: (b.entities[0], b.entities[0]), "unique"),
        (
            "relations",
            lambda b: (replace(b.relations[0], target_entity_key="e" * 64),),
            "endpoint",
        ),
        ("evidence", lambda b: (replace(b.evidence[0], chunk_number=3),), "chunk"),
        (
            "evidence",
            lambda b: (replace(b.evidence[0], source_document_key="e" * 64),),
            "signature",
        ),
        (
            "artifact_provenance",
            lambda b: (replace(b.artifact_provenance[0], artifact_key="e" * 64),),
            "provenance",
        ),
        ("counts", lambda b: replace(b.counts, entity_count=3), "count"),
    ],
)
def test_bundle_rejects_order_duplicates_broken_closure_and_bad_counts(
    field, value, message
) -> None:
    bundle = _bundle()
    with pytest.raises(ValueError, match=message):
        replace(bundle, **{field: value(bundle)})


@pytest.mark.parametrize(
    ("factory", "error"),
    [
        (lambda: replace(_bundle().counts, entity_count=True), (TypeError, ValueError)),
        (lambda: replace(_bundle().entities[0], entity_key="A" * 64), ValueError),
        (lambda: replace(_bundle().entities[0], cluster_key="cluster-a"), ValueError),
        (lambda: replace(_bundle().generation, schema_version=""), ValueError),
        (
            lambda: PrivateProjectionChunkReferenceV1(
                K["chunk"], True, DOCUMENT_UUID, 2
            ),
            (TypeError, ValueError),
        ),
        (
            lambda: PrivateProjectionChunkReferenceV1(
                K["chunk"], 1, "AAAAAAAA-1234-5678-9234-567812345678", 2
            ),
            ValueError,
        ),
        (
            lambda: replace(
                _bundle().artifact_provenance[0],
                scope_type=type("_StringSubclass", (str,), {})("collection"),
            ),
            TypeError,
        ),
    ],
)
def test_records_reject_bool_bad_digest_token_and_noncanonical_uuid(factory, error):
    with pytest.raises(error):
        factory()
