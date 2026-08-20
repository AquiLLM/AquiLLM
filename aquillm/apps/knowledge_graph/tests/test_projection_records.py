from dataclasses import FrozenInstanceError, replace

import pytest

from apps.knowledge_graph.projection.records import (
    AutomaticCanonicalMembershipV1,
    CollectionGraphProjectionBundleV1,
    ProjectedArtifactProvenanceV1,
    ProjectedChunkMembershipV1,
    ProjectedDocumentMembershipV1,
    ProjectedEntityV1,
    ProjectedPhysicalRelationV1,
    ProjectedRelationEvidenceV1,
    ProjectionCountsV1,
    ProjectionGenerationMarkerV1,
)

_NAMES = (
    "generation collection artifact entity_a entity_b document chunk relation "
    "evidence mention provenance automatic scope build rebuild document_artifact"
).split()
K = dict(zip(_NAMES, (character * 64 for character in "0123456789abcdef"), strict=True))
DIGEST = "f" * 64


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
        AutomaticCanonicalMembershipV1(
            entity_key=K["entity_b"],
            automatic_membership_key=K["automatic"],
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
            artifact_key=K["document_artifact"],
            source_document_key=K["document"],
            head_mention_key=K["mention"],
            tail_mention_key=K["evidence"],
            head_mapping_key=K["entity_a"],
            tail_mapping_key=K["entity_b"],
            orientation="head_to_tail",
            relation_type="knows",
            ontology_checksum=DIGEST,
            assembly_config_checksum=DIGEST,
            provenance_key=K["provenance"],
            semantic_signature=DIGEST,
        ),
    )
    collection_provenance = ProjectedArtifactProvenanceV1(
        artifact_key=K["artifact"],
        scope_type="collection",
        scope_key=K["collection"],
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
    )
    provenance = (
        collection_provenance,
        replace(
            collection_provenance,
            artifact_key=K["document_artifact"],
            scope_type="document",
            scope_key=K["document"],
            embedding_model_signature="",
            assembly_version="not-applicable",
            assembly_config_checksum="e" * 64,
        ),
    )
    counts = ProjectionCountsV1(2, 2, 1, 1, 1, 1, 2)
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


def _replace_row(rows, index=0, **changes):
    return rows[:index] + (replace(rows[index], **changes),) + rows[index + 1 :]


def _row_mutation(name, index=0, **changes):
    def mutate(bundle):
        rows = _replace_row(getattr(bundle, name), index, **changes)
        return replace(bundle, **{name: rows})

    return mutate


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda b: replace(b, entities=tuple(reversed(b.entities))), "sorted"),
        (lambda b: replace(b, entities=(b.entities[0],) * 2), "unique"),
        (_row_mutation("relations", target_entity_key="e" * 64), "endpoint"),
        (_row_mutation("evidence", chunk_number=3), "evidence"),
        (_row_mutation("evidence", source_document_key="e" * 64), "evidence"),
        (
            lambda b: replace(b, counts=replace(b.counts, entity_count=3)),
            "count",
        ),
        (
            lambda b: replace(
                b,
                automatic_memberships=b.automatic_memberships[:-1],
                counts=replace(b.counts, automatic_membership_count=1),
            ),
            "membership",
        ),
        (
            _row_mutation("automatic_memberships", decision_checksum="e" * 64),
            "membership",
        ),
        (
            _row_mutation("automatic_memberships", resolver_version="resolver-v2"),
            "membership",
        ),
        (
            lambda b: replace(
                b,
                artifact_provenance=b.artifact_provenance[1:],
                counts=replace(b.counts, artifact_provenance_count=1),
            ),
            "collection provenance",
        ),
        (_row_mutation("evidence", artifact_key="e" * 64), "evidence"),
        (_row_mutation("evidence", relation_type="likes"), "evidence"),
        (_row_mutation("evidence", ontology_checksum="e" * 64), "evidence"),
        (_row_mutation("evidence", assembly_config_checksum="e" * 64), "evidence"),
        (
            lambda b: replace(
                b,
                artifact_provenance=_replace_row(
                    b.artifact_provenance, 1, assembly_config_checksum="e" * 64
                ),
                evidence=_replace_row(b.evidence, assembly_config_checksum="e" * 64),
            ),
            "evidence",
        ),
    ],
)
def test_bundle_rejects_membership_provenance_and_evidence_incoherence(
    mutation, message
) -> None:
    with pytest.raises(ValueError, match=message):
        mutation(_bundle())


@pytest.mark.parametrize(
    "changes",
    [
        {"resolver_version": "resolver-v2"},
        {"resolution_config_checksum": "e" * 64},
        {"ontology_version": "ontology-v2"},
        {"ontology_checksum": "e" * 64},
        {"filter_policy_version": "filter-v2"},
        {"filter_policy_checksum": "e" * 64},
        {"extractor_version": "extractor-v2"},
        {"orchestration_version": 5},
    ],
)
def test_bundle_rejects_shared_provenance_identity_drift(changes) -> None:
    bundle = _bundle()
    rows = _replace_row(bundle.artifact_provenance, 1, **changes)
    with pytest.raises(ValueError, match="shared provenance"):
        replace(bundle, artifact_provenance=rows)


def test_document_provenance_allows_empty_embedding_signature() -> None:
    document = _bundle().artifact_provenance[1]
    assert document.embedding_model_signature == ""
    with pytest.raises(ValueError, match="document embedding"):
        replace(document, embedding_model_signature="embed-v1")
    with pytest.raises(ValueError, match="collection embedding"):
        replace(_bundle().artifact_provenance[0], embedding_model_signature="")
    with pytest.raises(TypeError, match="built-in str"):
        replace(
            document,
            embedding_model_signature=type("_SignatureSubclass", (str,), {})(""),
        )


def test_evidence_assembly_uses_collection_not_document_provenance() -> None:
    bundle = _bundle()
    evidence = bundle.evidence[0]
    assert (
        evidence.assembly_config_checksum
        == bundle.artifact_provenance[0].assembly_config_checksum
    )
    assert (
        evidence.assembly_config_checksum
        != bundle.artifact_provenance[1].assembly_config_checksum
    )


def test_bundle_rejects_chunk_key_and_coordinate_conflicts_independently() -> None:
    bundle = _bundle()
    original = bundle.chunks[0]
    for duplicate in (
        replace(original, chunk_number=3),
        replace(original, chunk_key="e" * 64),
    ):
        rows = tuple(
            sorted(
                (original, duplicate),
                key=lambda row: (row.document_key, row.chunk_number, row.chunk_key),
            )
        )
        with pytest.raises(ValueError, match="chunk"):
            replace(bundle, chunks=rows, counts=replace(bundle.counts, chunk_count=2))
