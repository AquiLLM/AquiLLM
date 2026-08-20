# ruff: noqa: E501
from dataclasses import FrozenInstanceError, fields, replace
from hashlib import sha256

import pytest

from apps.knowledge_graph.projection.records import PrivateProjectionChunkReferenceV1
from apps.knowledge_graph.retrieval.projected_types import (
    ProjectedAlgorithmSignatureV1,
    ProjectedAllowedScopeV1,
    ProjectedArtifactProvenanceV1,
    ProjectedAuthorizedGraphSnapshotV1,
    ProjectedAutomaticMembershipAuditV1,
    ProjectedChunkEvidenceV1,
    ProjectedEvidenceOrientationV1,
    ProjectedEvidenceSignatureV1,
    ProjectedFallbackMentionAuditV1,
    ProjectedIdentityMentionV1,
    ProjectedPhysicalRelationAuditV1,
    ProjectedRelationEvidenceAuditV1,
    ProjectedRelationGroupV1,
    ProjectedRetrievalDirectionV1,
    ProjectedScopeTypeV1,
    ProjectedSeedIdentityV1,
    ProjectedSnapshotCapsV1,
    canonical_projected_snapshot_bytes,
    projected_snapshot_checksum,
)


def _key(character: str) -> str:
    return character * 64


K = {character: _key(character) for character in "0123456789abcdef"}


def _provenance(scope_type: ProjectedScopeTypeV1) -> ProjectedArtifactProvenanceV1:
    is_collection = scope_type is ProjectedScopeTypeV1.COLLECTION
    return ProjectedArtifactProvenanceV1(
        artifact_key=K["a"] if is_collection else K["b"],
        scope_type=scope_type,
        scope_key=K["c"] if is_collection else K["d"],
        collection_key=K["c"],
        rebuild_request_key=None,
        evaluation_only=False,
        build_key=K["e" if is_collection else "f"],
        build_generation=3,
        orchestration_version=1,
        source_hash=K["0"],
        ontology_version="ontology-v1",
        ontology_checksum=K["1"],
        extractor_version="extractor-v1",
        resolver_version="resolver-v1",
        resolution_config_checksum=K["2"],
        filter_policy_version="filter-v1",
        filter_policy_checksum=K["3"],
        embedding_model_signature="embed-v1" if is_collection else "",
        assembly_version="assembly-v1",
        assembly_config_checksum=K["4"],
    )


def _signature() -> ProjectedEvidenceSignatureV1:
    return ProjectedEvidenceSignatureV1(
        evidence_key=K["5"],
        relation_key=K["6"],
        relation_mention_key=K["7"],
        chunk_key=K["8"],
        document_key=K["d"],
        chunk_number=2,
        confidence=0.75,
        artifact_key=K["b"],
        source_document_key=K["d"],
        head_mention_key=K["9"],
        tail_mention_key=K["a"],
        relation_type="works_at",
        head_mapping_key=K["b"],
        tail_mapping_key=K["c"],
        orientation=ProjectedEvidenceOrientationV1.HEAD_TO_TAIL,
        ontology_checksum=K["1"],
        assembly_config_checksum=K["4"],
    )


def _chunk() -> ProjectedChunkEvidenceV1:
    signature = _signature()
    return ProjectedChunkEvidenceV1(
        chunk_key=signature.chunk_key,
        document_key=signature.document_key,
        chunk_number=signature.chunk_number,
        confidence=signature.confidence,
        provenance_key=signature.evidence_key,
    )


def _snapshot(*, graph: bool = True) -> ProjectedAuthorizedGraphSnapshotV1:
    memberships = (
        ProjectedAutomaticMembershipAuditV1(0, K["9"], K["1"], K["2"], "resolver-v1"),
        ProjectedAutomaticMembershipAuditV1(0, K["a"], K["2"], K["2"], "resolver-v1"),
    )
    audit_rows: tuple[object, ...] = memberships
    groups: tuple[ProjectedRelationGroupV1, ...] = ()
    if graph:
        physical = ProjectedPhysicalRelationAuditV1(
            1, K["6"], K["a"], K["9"], "works_at", K["a"]
        )
        relation_evidence = ProjectedRelationEvidenceAuditV1(1, _signature())
        groups = (
            ProjectedRelationGroupV1(
                K["1"],
                "works_at",
                K["2"],
                ProjectedRetrievalDirectionV1.FORWARD,
                0.5,
                1,
                (_chunk(),),
            ),
        )
        audit_rows += (physical, relation_evidence)
    return ProjectedAuthorizedGraphSnapshotV1(
        algorithm=ProjectedAlgorithmSignatureV1(
            "ppr_projected_v1",
            "ppr_transition_v1",
            "ppr_evidence_v1",
            "rrf_seed_v1",
            K["f"],
        ),
        caps=ProjectedSnapshotCapsV1(64, 10_000, 128, 2, 200, 1_000, 3_000, 3, 2),
        load_max_hops=2,
        allowed_scope=ProjectedAllowedScopeV1((K["d"],), (K["c"],), K["e"]),
        identity_keys=(K["1"], K["2"]),
        seed_identities=(ProjectedSeedIdentityV1(K["8"], K["1"]),),
        relation_groups=groups,
        mentions=(),
        artifact_provenance=(
            _provenance(ProjectedScopeTypeV1.COLLECTION),
            _provenance(ProjectedScopeTypeV1.DOCUMENT),
        ),
        audit_rows=audit_rows,
    )


def test_exact_fields_and_closed_audit_tags_are_frozen() -> None:
    assert tuple(field.name for field in fields(ProjectedEvidenceSignatureV1)) == (
        "evidence_key",
        "relation_key",
        "relation_mention_key",
        "chunk_key",
        "document_key",
        "chunk_number",
        "confidence",
        "artifact_key",
        "source_document_key",
        "head_mention_key",
        "tail_mention_key",
        "relation_type",
        "head_mapping_key",
        "tail_mapping_key",
        "orientation",
        "ontology_checksum",
        "assembly_config_checksum",
    )
    assert tuple(field.name for field in fields(ProjectedArtifactProvenanceV1)) == (
        "artifact_key",
        "scope_type",
        "scope_key",
        "collection_key",
        "rebuild_request_key",
        "evaluation_only",
        "build_key",
        "build_generation",
        "orchestration_version",
        "source_hash",
        "ontology_version",
        "ontology_checksum",
        "extractor_version",
        "resolver_version",
        "resolution_config_checksum",
        "filter_policy_version",
        "filter_policy_checksum",
        "embedding_model_signature",
        "assembly_version",
        "assembly_config_checksum",
    )
    expected = {
        ProjectedAutomaticMembershipAuditV1: (
            "discovery_hop",
            "entity_key",
            "automatic_membership_key",
            "decision_checksum",
            "resolver_version",
            "kind",
        ),
        ProjectedPhysicalRelationAuditV1: (
            "discovery_hop",
            "relation_key",
            "artifact_key",
            "source_entity_key",
            "relation_type",
            "target_entity_key",
            "kind",
        ),
        ProjectedRelationEvidenceAuditV1: ("discovery_hop", "signature", "kind"),
        ProjectedFallbackMentionAuditV1: (
            "discovery_hop",
            "identity_key",
            "evidence",
            "kind",
        ),
    }
    for kind, names in expected.items():
        assert tuple(field.name for field in fields(kind)) == names
    # fmt: off
    assert tuple(field.name for field in fields(ProjectedChunkEvidenceV1)) == ("chunk_key", "document_key", "chunk_number", "confidence", "provenance_key")
    assert tuple(field.name for field in fields(ProjectedSeedIdentityV1)) == ("seed_chunk_key", "identity_key")
    assert tuple(field.name for field in fields(ProjectedRelationGroupV1)) == ("source_identity_key", "relation_type", "target_identity_key", "direction", "raw_weight", "admission_hop", "evidence")
    assert tuple(field.name for field in fields(ProjectedIdentityMentionV1)) == ("identity_key", "evidence")
    assert tuple(field.name for field in fields(ProjectedAllowedScopeV1)) == ("document_keys", "collection_keys", "scope_version_signature")
    assert tuple(field.name for field in fields(ProjectedAlgorithmSignatureV1)) == ("algorithm_version", "transition_version", "evidence_version", "seed_version", "algorithm_signature")
    assert tuple(field.name for field in fields(ProjectedSnapshotCapsV1)) == ("max_seeds", "max_scope_documents", "max_scope_collections", "max_hops", "max_nodes", "max_edges", "max_evidence_rows", "max_evidence_per_edge", "max_mentions_per_entity")
    assert tuple(field.name for field in fields(ProjectedAuthorizedGraphSnapshotV1)) == ("algorithm", "caps", "load_max_hops", "allowed_scope", "identity_keys", "seed_identities", "relation_groups", "mentions", "artifact_provenance", "audit_rows")
    assert tuple(ProjectedRetrievalDirectionV1) == ("forward", "reverse_directed", "undirected")
    # fmt: on
    snapshot = _snapshot()
    assert tuple(row.kind for row in snapshot.audit_rows) == (
        "automatic_membership",
        "automatic_membership",
        "physical_relation",
        "relation_evidence",
    )
    with pytest.raises(FrozenInstanceError):
        snapshot.load_max_hops = 1
    assert not hasattr(snapshot, "__dict__")


def test_strict_opaque_types_numbers_enums_and_subclasses_are_rejected() -> None:
    class Text(str):
        pass

    for value in (17, K["a"].upper(), Text(K["5"]), "f" * 63):
        with pytest.raises((TypeError, ValueError)):
            replace(_signature(), evidence_key=value)
    with pytest.raises(TypeError):
        replace(_signature(), relation_type=Text("works_at"))
    for value in (True, -1, 2**31):
        with pytest.raises((TypeError, ValueError)):
            replace(_signature(), chunk_number=value)
    for value in (1, float("nan"), float("inf"), -0.1, 1.1):
        with pytest.raises((TypeError, ValueError)):
            replace(_signature(), confidence=value)
    with pytest.raises((TypeError, ValueError)):
        replace(_signature(), orientation="head_to_tail")
    with pytest.raises((TypeError, ValueError)):
        replace(_snapshot().caps, max_nodes=True)


def test_order_caps_and_all_snapshot_closures_are_enforced() -> None:
    snapshot = _snapshot()
    mutations = (
        {"identity_keys": tuple(reversed(snapshot.identity_keys))},
        {"identity_keys": (K["1"], K["1"])},
        {"seed_identities": (ProjectedSeedIdentityV1(K["8"], K["3"]),)},
        {"relation_groups": snapshot.relation_groups * 2},
        {"audit_rows": tuple(reversed(snapshot.audit_rows))},
        {"artifact_provenance": tuple(reversed(snapshot.artifact_provenance))},
        {"allowed_scope": ProjectedAllowedScopeV1((K["e"],), (K["c"],), K["e"])},
        {"audit_rows": snapshot.audit_rows[:2] + snapshot.audit_rows[3:]},
        {"audit_rows": snapshot.audit_rows[:1] + snapshot.audit_rows[2:]},
        {"caps": replace(snapshot.caps, max_nodes=1)},
        {"load_max_hops": 0},
    )
    for mutation in mutations:
        with pytest.raises((TypeError, ValueError)):
            replace(snapshot, **mutation)
    mention = ProjectedIdentityMentionV1(K["1"], _chunk())
    with pytest.raises(ValueError, match="fallback"):
        replace(snapshot, mentions=(mention,))


def test_canonical_snapshot_vector_and_private_integer_canary_exclusion() -> None:
    snapshot = _snapshot(graph=False)
    encoded = canonical_projected_snapshot_bytes(snapshot)
    assert encoded.startswith(b'{"algorithm":{"algorithm_signature":"ffffffff')
    assert b'"confidence":"0x1.8000000000000p-1"' in canonical_projected_snapshot_bytes(
        _snapshot()
    )
    assert b'"audit_rows":[{"automatic_membership_key":"1111' in encoded
    assert projected_snapshot_checksum(snapshot) == sha256(encoded).hexdigest()
    assert projected_snapshot_checksum(snapshot) == (
        "8f2cbb9d65cd88971eb31c0787ab29201bd298df704eb310cf8d4fee38bfc277"
    )
    private = PrivateProjectionChunkReferenceV1(
        K["8"], 9_223_372_036_854_775_807, "12345678-1234-5678-9234-567812345678", 2
    )
    with pytest.raises(TypeError):
        replace(snapshot, mentions=(private,))
    with pytest.raises(TypeError):
        canonical_projected_snapshot_bytes(private)
    assert b"9223372036854775807" not in encoded
