# ruff: noqa: E501, E701, E702
from dataclasses import FrozenInstanceError, fields, replace
from hashlib import sha256
from typing import Literal, get_type_hints

import pytest

from apps.knowledge_graph.projection.records import PrivateProjectionChunkReferenceV1
from apps.knowledge_graph.retrieval.projected_types import (
    ProjectedAlgorithmSignatureV1,
    ProjectedAllowedScopeV1,
    ProjectedArtifactProvenanceV1,
    ProjectedAuditRowV1,
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
EXPECTED_SNAPSHOT_BYTES = b'{"algorithm":{"algorithm_signature":"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff","algorithm_version":"ppr_projected_v1","evidence_version":"ppr_evidence_v1","seed_version":"rrf_seed_v1","transition_version":"ppr_transition_v1"},"allowed_scope":{"collection_keys":["cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"],"document_keys":["dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"],"scope_version_signature":"eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"},"artifact_provenance":[{"artifact_key":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","assembly_config_checksum":"4444444444444444444444444444444444444444444444444444444444444444","assembly_version":"assembly-v1","build_generation":3,"build_key":"eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee","collection_key":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc","embedding_model_signature":"embed-v1","evaluation_only":false,"extractor_version":"extractor-v1","filter_policy_checksum":"3333333333333333333333333333333333333333333333333333333333333333","filter_policy_version":"filter-v1","ontology_checksum":"1111111111111111111111111111111111111111111111111111111111111111","ontology_version":"ontology-v1","orchestration_version":1,"rebuild_request_key":null,"resolution_config_checksum":"2222222222222222222222222222222222222222222222222222222222222222","resolver_version":"resolver-v1","scope_key":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc","scope_type":"collection","source_hash":"0000000000000000000000000000000000000000000000000000000000000000"},{"artifact_key":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","assembly_config_checksum":"4444444444444444444444444444444444444444444444444444444444444444","assembly_version":"assembly-v1","build_generation":3,"build_key":"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff","collection_key":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc","embedding_model_signature":"","evaluation_only":false,"extractor_version":"extractor-v1","filter_policy_checksum":"3333333333333333333333333333333333333333333333333333333333333333","filter_policy_version":"filter-v1","ontology_checksum":"1111111111111111111111111111111111111111111111111111111111111111","ontology_version":"ontology-v1","orchestration_version":1,"rebuild_request_key":null,"resolution_config_checksum":"2222222222222222222222222222222222222222222222222222222222222222","resolver_version":"resolver-v1","scope_key":"dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd","scope_type":"document","source_hash":"0000000000000000000000000000000000000000000000000000000000000000"}],"audit_rows":[{"automatic_membership_key":"1111111111111111111111111111111111111111111111111111111111111111","decision_checksum":"2222222222222222222222222222222222222222222222222222222222222222","discovery_hop":0,"entity_key":"9999999999999999999999999999999999999999999999999999999999999999","kind":"automatic_membership","resolver_version":"resolver-v1"},{"automatic_membership_key":"2222222222222222222222222222222222222222222222222222222222222222","decision_checksum":"2222222222222222222222222222222222222222222222222222222222222222","discovery_hop":0,"entity_key":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","kind":"automatic_membership","resolver_version":"resolver-v1"}],"caps":{"max_edges":1000,"max_evidence_per_edge":3,"max_evidence_rows":3000,"max_hops":2,"max_mentions_per_entity":2,"max_nodes":200,"max_scope_collections":128,"max_scope_documents":10000,"max_seeds":64},"identity_keys":["1111111111111111111111111111111111111111111111111111111111111111","2222222222222222222222222222222222222222222222222222222222222222"],"load_max_hops":2,"mentions":[],"relation_groups":[],"seed_identities":[{"identity_key":"1111111111111111111111111111111111111111111111111111111111111111","seed_chunk_key":"8888888888888888888888888888888888888888888888888888888888888888"}]}'


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
    # fmt: off
    assert tuple(field.name for field in fields(ProjectedEvidenceSignatureV1)) == ("evidence_key", "relation_key", "relation_mention_key", "chunk_key", "document_key", "chunk_number", "confidence", "artifact_key", "source_document_key", "head_mention_key", "tail_mention_key", "relation_type", "head_mapping_key", "tail_mapping_key", "orientation", "ontology_checksum", "assembly_config_checksum")
    assert tuple(field.name for field in fields(ProjectedArtifactProvenanceV1)) == ("artifact_key", "scope_type", "scope_key", "collection_key", "rebuild_request_key", "evaluation_only", "build_key", "build_generation", "orchestration_version", "source_hash", "ontology_version", "ontology_checksum", "extractor_version", "resolver_version", "resolution_config_checksum", "filter_policy_version", "filter_policy_checksum", "embedding_model_signature", "assembly_version", "assembly_config_checksum")
    expected = {ProjectedAutomaticMembershipAuditV1: ("discovery_hop", "entity_key", "automatic_membership_key", "decision_checksum", "resolver_version", "kind"), ProjectedPhysicalRelationAuditV1: ("discovery_hop", "relation_key", "artifact_key", "source_entity_key", "relation_type", "target_entity_key", "kind"), ProjectedRelationEvidenceAuditV1: ("discovery_hop", "signature", "kind"), ProjectedFallbackMentionAuditV1: ("discovery_hop", "identity_key", "evidence", "kind")}
    for kind, names in expected.items():
        assert tuple(field.name for field in fields(kind)) == names
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


# fmt: off
def test_reviewed_audit_caps_and_provenance_roles_are_closed() -> None:
    base = _snapshot(graph=False); collapsed = replace(base.audit_rows[1], automatic_membership_key=K["1"])
    with pytest.raises(ValueError, match="automatic|audit"): replace(base, caps=replace(base.caps, max_nodes=1), identity_keys=(K["1"],), audit_rows=(base.audit_rows[0], collapsed))
    snapshot = _snapshot(); physical = snapshot.audit_rows[2]; evidence = snapshot.audit_rows[3]
    signature2 = replace(_signature(), evidence_key=K["0"], relation_key=K["7"], relation_mention_key=K["0"], chunk_key=K["0"]); evidence2 = ProjectedRelationEvidenceAuditV1(1, signature2)
    physical2 = replace(physical, relation_key=K["7"]); chunk2 = replace(_chunk(), chunk_key=K["0"], provenance_key=K["0"]); group = replace(snapshot.relation_groups[0], evidence=(chunk2, _chunk()))
    audits = (*snapshot.audit_rows[:2], physical, physical2, evidence2, evidence)
    with pytest.raises(ValueError, match="physical|audit"): replace(snapshot, caps=replace(snapshot.caps, max_edges=1), relation_groups=(group,), audit_rows=audits)
    reverse2 = replace(snapshot.relation_groups[0], source_identity_key=K["2"], target_identity_key=K["1"], direction=ProjectedRetrievalDirectionV1.REVERSE_DIRECTED, evidence=(chunk2,))
    with pytest.raises(ValueError, match="evidence|audit"): replace(snapshot, caps=replace(snapshot.caps, max_evidence_rows=1, max_evidence_per_edge=1), relation_groups=(replace(snapshot.relation_groups[0], evidence=(_chunk(),)), reverse2), audit_rows=audits)
    mutations = (replace(physical, artifact_key=K["b"]), replace(evidence, signature=replace(_signature(), artifact_key=K["a"])), replace(evidence, signature=replace(_signature(), relation_type="reports_to")), replace(evidence, signature=replace(_signature(), ontology_checksum=K["2"])), replace(evidence, signature=replace(_signature(), assembly_config_checksum=K["3"])))
    for mutation in mutations:
        with pytest.raises(ValueError): replace(snapshot, audit_rows=(*snapshot.audit_rows[:2], mutation if type(mutation) is ProjectedPhysicalRelationAuditV1 else physical, mutation) if type(mutation) is ProjectedRelationEvidenceAuditV1 else (*snapshot.audit_rows[:2], mutation, evidence))
    distinct_document = replace(snapshot.artifact_provenance[1], ontology_checksum=K["5"], assembly_config_checksum=K["6"])
    replace(snapshot, artifact_provenance=(snapshot.artifact_provenance[0], distinct_document))
def test_reviewed_strict_values_public_types_and_runtime_finality() -> None:
    class Text(str): pass
    for value in (None, False, Text("")):
        with pytest.raises((TypeError, ValueError)): replace(_provenance(ProjectedScopeTypeV1.DOCUMENT), embedding_model_signature=value)
    for value in ("", None, False, Text("embed-v1")):
        with pytest.raises((TypeError, ValueError)): replace(_provenance(ProjectedScopeTypeV1.COLLECTION), embedding_model_signature=value)
    for value in ("WorksAt", "works-at", "a" * 129):
        for row in (_signature(), _snapshot().relation_groups[0], _snapshot().audit_rows[2]):
            with pytest.raises(ValueError, match="relation_type"): replace(row, relation_type=value)
    algorithm = _snapshot().algorithm
    for name in ("algorithm_version", "transition_version", "evidence_version", "seed_version"):
        with pytest.raises(ValueError, match=name): replace(algorithm, **{name: "wrong-v1"})
    assert get_type_hints(ProjectedAuthorizedGraphSnapshotV1)["audit_rows"] == tuple[ProjectedAuditRowV1, ...]
    tags = ((ProjectedAutomaticMembershipAuditV1, "automatic_membership"), (ProjectedPhysicalRelationAuditV1, "physical_relation"), (ProjectedRelationEvidenceAuditV1, "relation_evidence"), (ProjectedFallbackMentionAuditV1, "fallback_mention"))
    for kind, tag in tags: assert get_type_hints(kind)["kind"] == Literal[tag]
    with pytest.raises(TypeError, match="final"):
        class InvalidProjectedSubclass(ProjectedChunkEvidenceV1): pass
def test_evidence_reference_directions_and_fallback_semantics_are_bounded() -> None:
    snapshot = _snapshot(); forward = snapshot.relation_groups[0]
    undirected = replace(forward, direction=ProjectedRetrievalDirectionV1.UNDIRECTED); reverse = replace(forward, source_identity_key=K["2"], target_identity_key=K["1"], direction=ProjectedRetrievalDirectionV1.REVERSE_DIRECTED)
    with pytest.raises(ValueError, match="evidence.*reference|direction"): replace(snapshot, relation_groups=(forward, undirected, reverse))
    with pytest.raises(ValueError, match="direction"): replace(snapshot, relation_groups=(forward, undirected))
    replace(snapshot, relation_groups=(forward, reverse))
    replace(snapshot, relation_groups=(undirected, replace(undirected, source_identity_key=K["2"], target_identity_key=K["1"])))
    base = _snapshot(graph=False); first = replace(_chunk(), confidence=0.5); second = _chunk()
    fallbacks = (ProjectedFallbackMentionAuditV1(0, K["1"], first), ProjectedFallbackMentionAuditV1(0, K["1"], second))
    with pytest.raises(ValueError, match="fallback.*duplicate|semantic"): replace(base, mentions=(ProjectedIdentityMentionV1(K["1"], second),), audit_rows=(*base.audit_rows, *fallbacks))
    conflict = replace(second, provenance_key=K["6"], chunk_number=3); conflict_rows = (ProjectedFallbackMentionAuditV1(0, K["1"], second), ProjectedFallbackMentionAuditV1(0, K["1"], conflict))
    with pytest.raises(ValueError, match="coordinate"): replace(base, mentions=(ProjectedIdentityMentionV1(K["1"], second), ProjectedIdentityMentionV1(K["1"], conflict)), audit_rows=(*base.audit_rows, *conflict_rows))
def test_scope_cardinality_discovery_hops_and_seed_caps_are_complete() -> None:
    snapshot = _snapshot(); collection, document = snapshot.artifact_provenance
    duplicate_collection = replace(collection, artifact_key=K["0"], build_key=K["0"]); duplicate_document = replace(document, artifact_key=K["0"], build_key=K["0"])
    for rows in ((duplicate_collection, collection, document), (collection, duplicate_document, document)):
        with pytest.raises(ValueError, match="scope|provenance"): replace(snapshot, artifact_provenance=rows)
    collection2 = replace(collection, artifact_key=K["0"], scope_key=K["e"], collection_key=K["e"], build_key=K["0"]); document1 = replace(document, collection_key=K["e"]); document2 = replace(document, artifact_key=K["5"], scope_key=K["f"], collection_key=K["e"], build_key=K["5"])
    with pytest.raises(ValueError, match="collection|provenance"): replace(snapshot, allowed_scope=ProjectedAllowedScopeV1((K["d"], K["f"]), (K["c"], K["e"]), K["e"]), artifact_provenance=(collection, collection2, document1, document2))
    with pytest.raises(ValueError, match="hop"): replace(snapshot, audit_rows=(*snapshot.audit_rows[:3], replace(snapshot.audit_rows[3], discovery_hop=2)))
    raw = _snapshot(graph=False); collapsed = replace(raw.audit_rows[1], automatic_membership_key=K["1"], discovery_hop=1)
    base = replace(raw, identity_keys=(K["1"],), audit_rows=(raw.audit_rows[0], collapsed)); fallback = ProjectedFallbackMentionAuditV1(1, K["1"], _chunk())
    with pytest.raises(ValueError, match="hop"): replace(base, mentions=(ProjectedIdentityMentionV1(K["1"], _chunk()),), audit_rows=(*base.audit_rows, fallback))
    replace(snapshot, caps=replace(snapshot.caps, max_seeds=1), seed_identities=(ProjectedSeedIdentityV1(K["8"], K["1"]), ProjectedSeedIdentityV1(K["8"], K["2"])))
    with pytest.raises(ValueError, match="seed"): replace(snapshot, caps=replace(snapshot.caps, max_seeds=1), seed_identities=(ProjectedSeedIdentityV1(K["7"], K["1"]), ProjectedSeedIdentityV1(K["8"], K["2"])))
    with pytest.raises(ValueError, match="seed|cap"): replace(snapshot, caps=replace(snapshot.caps, max_seeds=2, max_nodes=2), seed_identities=(ProjectedSeedIdentityV1(K["7"], K["1"]), ProjectedSeedIdentityV1(K["8"], K["1"]), ProjectedSeedIdentityV1(K["8"], K["2"])))
# fmt: on


def test_canonical_snapshot_vector_and_private_integer_canary_exclusion() -> None:
    snapshot = _snapshot(graph=False)
    encoded = canonical_projected_snapshot_bytes(snapshot)
    assert encoded == EXPECTED_SNAPSHOT_BYTES
    assert b'"confidence":"0x1.8000000000000p-1"' in canonical_projected_snapshot_bytes(
        _snapshot()
    )
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
