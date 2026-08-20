# ruff: noqa: E501, E701, E702
import ast
from dataclasses import FrozenInstanceError, fields, replace
from hashlib import sha256
from pathlib import Path
from typing import Literal, get_type_hints

import pytest

from apps.knowledge_graph.projection.records import PrivateProjectionChunkReferenceV1
from apps.knowledge_graph.retrieval import projected_types as projected_types_module
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

# fmt: off
K = {"algorithm_signature": "05217ef971fa843650d0920af2ee21dc07725823c0104dcc297a1e743f473e43", "scope_signature": "153a16acae9f259594b8b58337c325e1894e3bdd8cf5a4ac1907b530bcac6525", "collection_a": "3cdb46e706e9c01df159c90ca6464394bb912b3b0acf5ba3ec84bb0fb2070320", "document_a": "7401b5f48372c8ea702fa477f6e3f02604825aee3d4146401ea698141e188c57", "collection_artifact_a": "699b363e2d584502f5c2d84dffb5fa0f748190f71048bc862ed5d7069e12d68a", "document_artifact_a": "3ba75fda00841f7e904cdc4ae170ae966be2fa1881996051e9727b9daa0be101", "build_collection_a": "7707a57f1324bb446309462c7939d03c77f06ad030ab879b664093fa1d2eb820", "build_document_a": "fd267eca6c38095c65b1774750b58e8c98de74534fa4997f3c7940284d1fcacb", "source_collection_a": "0589da0d55c1e138ade7e4e516c6c2fca72378e89ec97b3430d88f77e6e4819e", "source_document_a": "c8d0c6ba015b29380b4ee6203436b03f884c2130efe1da64ebbf228d8a7c103b", "collection_ontology_checksum": "adf7cb191baab8c70ff56a867594fa526bd852570ceff98274e7c9d1476a143e", "document_ontology_checksum": "c55a6d91be1eae2247b4d8dae43cda44e762e67bb4cd140dab3077dd4b73091f", "collection_assembly_checksum": "6824b92fb9e0bad6b15ffd6b25647f080ac222201828746d320d6572fe58752c", "document_assembly_checksum": "6ad2af1d5d2901482b6cddceef8fa852f46d8bb1aef1e60553facc5fc03c2543", "collection_resolution_checksum": "f29f005d16a38375123d577c8a479607d2171d458d85ad016e655b2eb6f4d521", "document_resolution_checksum": "be60edf1f64767557d30ab49ab17874505a8185b87874cf88325e6ee53637d2a", "collection_filter_checksum": "9fb59889c21ae6988842014d203d4899b4af968924c375f336f3c9c260afff7e", "document_filter_checksum": "c23fdfd55af965ded049fa67890b9db0d07ffb3d1b0d013ae156bdb14a6bfe34", "evidence_a": "3c39a9f1301193cd705ae238b8cf2419d204995b4dcf9491fe2ade3cf2af1ffc", "evidence_b": "2863f57fd1b799a7c619eaf4b294fb562d1db40ffbda1cb8b28d9c01fa2539e8", "relation_a": "7a511eb902e62f29de2e34cb76b0eec18a82078ffb7e5272cb835ea344ff6354", "relation_b": "d26f5f6b68f2364d161128688e13df50253b536052b382f1402338f828fe1458", "relation_mention_a": "4c752989fbcd3d4396ddd46d473353056b9217057313cca2647ef14b9c5e24a0", "relation_mention_b": "774baa898b91f043290d8b068295bdf531b59bba8c68288c667b99137fa1436c", "chunk_a": "5e0840395f1be0c0dec5856a485f12312dc5367801c66b982e8e3e2013382eaf", "chunk_b": "093cb77b93ab8b4825671ac9e2182e5b9ed76a27b1c97118a6adba3b27b7d69c", "source_entity": "3da8a7117f999464846da3e2d532c3ae7ece4efd61c8697a7476f8f3772c0269", "target_entity": "c9b3e7b23b3e09638025b86a9a5630abb4c3b811dd0b98a2010e0335f5bb659c", "source_identity": "fa032f9010b262b3e139a70f27916af24ebd6a3158bf0204893c8ebbcc58e112", "target_identity": "ac43e82a633b092a515dd15ce3f767c9ab4cfb65bbb0d9ee4866264a2362c2ef", "head_mention_a": "b7f515c582638aa344ed333d0eeab209cb3c8edea38a97e8708677dd322d689e", "tail_mention_a": "5de43abb6c93e7625d2385715729899bb546881bafe9adf7887c1bb9f49b7b9d", "head_mapping_a": "900b6dd3fb6ad4e827c99d203c38ca9f275ae35d3e38fabb1112cfd6d9d49a4f", "tail_mapping_a": "82d7968a2e4949d16ce71fd38598ceff5a3fe71494adff99c2d981ce2faeb97b", "source_decision": "12fb259d7524a3184b4b3ea18ef29a3fde5af2c400e93cf1264e881574b6caa1", "target_decision": "cb47cdc80ff80c839a109e21419d95a7d4c42b986e0ff5a537352f54b3572558", "seed_chunk_a": "9f6a2a80839bfab8c2f951ab7af6389ff1c399b50c6505c902b718c939bd9483", "seed_chunk_b": "6d2c3ceae53e232eb9b9827ca5474e8b7636c15f77b42dec57418c13b558869e", "duplicate_artifact": "5c2d496549b9ea368d5dd6b7867ab57ccef838c8af917d68e4a1e9f8be6d6c07", "duplicate_build": "1cfda9c70be17a971792dda65c07c8bb97b4a3c958fcf7329062d9a7218c32c6", "duplicate_document": "75cd12daadbf8e23bb0b4fe3865a0e915006e95c149f1fc110bd21d8e198b588", "duplicate_collection": "2ce823b0bcdfa605fd4f18394f79d6ce93df26c2e03bac55acf91e3fac43402c", "other_identity": "a301c62fe9bed116221325ae21aefc5ded3e944125233acab37677b275a6b9fb", "conflicting_provenance": "30edfc955ed6d702d5a56207847d98a0d464c95fdd61ea607e0d7165c45a7db0"}
EXPECTED_SNAPSHOT_BYTES = b'{"algorithm":{"algorithm_signature":"05217ef971fa843650d0920af2ee21dc07725823c0104dcc297a1e743f473e43","algorithm_version":"ppr_projected_v1","evidence_version":"ppr_evidence_v1","seed_version":"rrf_seed_v1","transition_version":"ppr_transition_v1"},"allowed_scope":{"collection_keys":["3cdb46e706e9c01df159c90ca6464394bb912b3b0acf5ba3ec84bb0fb2070320"],"document_keys":["7401b5f48372c8ea702fa477f6e3f02604825aee3d4146401ea698141e188c57"],"scope_version_signature":"153a16acae9f259594b8b58337c325e1894e3bdd8cf5a4ac1907b530bcac6525"},"artifact_provenance":[{"artifact_key":"699b363e2d584502f5c2d84dffb5fa0f748190f71048bc862ed5d7069e12d68a","assembly_config_checksum":"6824b92fb9e0bad6b15ffd6b25647f080ac222201828746d320d6572fe58752c","assembly_version":"assembly-v1","build_generation":3,"build_key":"7707a57f1324bb446309462c7939d03c77f06ad030ab879b664093fa1d2eb820","collection_key":"3cdb46e706e9c01df159c90ca6464394bb912b3b0acf5ba3ec84bb0fb2070320","embedding_model_signature":"embed-v1","evaluation_only":false,"extractor_version":"extractor-v1","filter_policy_checksum":"9fb59889c21ae6988842014d203d4899b4af968924c375f336f3c9c260afff7e","filter_policy_version":"filter-v1","ontology_checksum":"adf7cb191baab8c70ff56a867594fa526bd852570ceff98274e7c9d1476a143e","ontology_version":"ontology-v1","orchestration_version":1,"rebuild_request_key":null,"resolution_config_checksum":"f29f005d16a38375123d577c8a479607d2171d458d85ad016e655b2eb6f4d521","resolver_version":"resolver-v1","scope_key":"3cdb46e706e9c01df159c90ca6464394bb912b3b0acf5ba3ec84bb0fb2070320","scope_type":"collection","source_hash":"0589da0d55c1e138ade7e4e516c6c2fca72378e89ec97b3430d88f77e6e4819e"},{"artifact_key":"3ba75fda00841f7e904cdc4ae170ae966be2fa1881996051e9727b9daa0be101","assembly_config_checksum":"6ad2af1d5d2901482b6cddceef8fa852f46d8bb1aef1e60553facc5fc03c2543","assembly_version":"assembly-v1","build_generation":3,"build_key":"fd267eca6c38095c65b1774750b58e8c98de74534fa4997f3c7940284d1fcacb","collection_key":"3cdb46e706e9c01df159c90ca6464394bb912b3b0acf5ba3ec84bb0fb2070320","embedding_model_signature":"","evaluation_only":false,"extractor_version":"extractor-v1","filter_policy_checksum":"c23fdfd55af965ded049fa67890b9db0d07ffb3d1b0d013ae156bdb14a6bfe34","filter_policy_version":"filter-v1","ontology_checksum":"c55a6d91be1eae2247b4d8dae43cda44e762e67bb4cd140dab3077dd4b73091f","ontology_version":"ontology-v1","orchestration_version":1,"rebuild_request_key":null,"resolution_config_checksum":"be60edf1f64767557d30ab49ab17874505a8185b87874cf88325e6ee53637d2a","resolver_version":"resolver-v1","scope_key":"7401b5f48372c8ea702fa477f6e3f02604825aee3d4146401ea698141e188c57","scope_type":"document","source_hash":"c8d0c6ba015b29380b4ee6203436b03f884c2130efe1da64ebbf228d8a7c103b"}],"audit_rows":[{"automatic_membership_key":"ac43e82a633b092a515dd15ce3f767c9ab4cfb65bbb0d9ee4866264a2362c2ef","decision_checksum":"cb47cdc80ff80c839a109e21419d95a7d4c42b986e0ff5a537352f54b3572558","discovery_hop":0,"entity_key":"c9b3e7b23b3e09638025b86a9a5630abb4c3b811dd0b98a2010e0335f5bb659c","kind":"automatic_membership","resolver_version":"resolver-v1"},{"automatic_membership_key":"fa032f9010b262b3e139a70f27916af24ebd6a3158bf0204893c8ebbcc58e112","decision_checksum":"12fb259d7524a3184b4b3ea18ef29a3fde5af2c400e93cf1264e881574b6caa1","discovery_hop":0,"entity_key":"3da8a7117f999464846da3e2d532c3ae7ece4efd61c8697a7476f8f3772c0269","kind":"automatic_membership","resolver_version":"resolver-v1"}],"caps":{"max_edges":1000,"max_evidence_per_edge":3,"max_evidence_rows":3000,"max_hops":2,"max_mentions_per_entity":2,"max_nodes":200,"max_scope_collections":128,"max_scope_documents":10000,"max_seeds":64},"identity_keys":["ac43e82a633b092a515dd15ce3f767c9ab4cfb65bbb0d9ee4866264a2362c2ef","fa032f9010b262b3e139a70f27916af24ebd6a3158bf0204893c8ebbcc58e112"],"load_max_hops":2,"mentions":[],"relation_groups":[],"seed_identities":[{"identity_key":"fa032f9010b262b3e139a70f27916af24ebd6a3158bf0204893c8ebbcc58e112","seed_chunk_key":"9f6a2a80839bfab8c2f951ab7af6389ff1c399b50c6505c902b718c939bd9483"}]}'


def _provenance(scope_type: ProjectedScopeTypeV1) -> ProjectedArtifactProvenanceV1:
    is_collection = scope_type is ProjectedScopeTypeV1.COLLECTION
    return ProjectedArtifactProvenanceV1(
        artifact_key=K["collection_artifact_a"] if is_collection else K["document_artifact_a"],
        scope_type=scope_type,
        scope_key=K["collection_a"] if is_collection else K["document_a"],
        collection_key=K["collection_a"],
        rebuild_request_key=None,
        evaluation_only=False,
        build_key=K["build_collection_a"] if is_collection else K["build_document_a"],
        build_generation=3,
        orchestration_version=1,
        source_hash=K["source_collection_a"] if is_collection else K["source_document_a"],
        ontology_version="ontology-v1",
        ontology_checksum=K["collection_ontology_checksum"] if is_collection else K["document_ontology_checksum"],
        extractor_version="extractor-v1",
        resolver_version="resolver-v1",
        resolution_config_checksum=K["collection_resolution_checksum"] if is_collection else K["document_resolution_checksum"],
        filter_policy_version="filter-v1",
        filter_policy_checksum=K["collection_filter_checksum"] if is_collection else K["document_filter_checksum"],
        embedding_model_signature="embed-v1" if is_collection else "",
        assembly_version="assembly-v1",
        assembly_config_checksum=K["collection_assembly_checksum"] if is_collection else K["document_assembly_checksum"],
    )


def _signature() -> ProjectedEvidenceSignatureV1:
    return ProjectedEvidenceSignatureV1(
        evidence_key=K["evidence_a"],
        relation_key=K["relation_a"],
        relation_mention_key=K["relation_mention_a"],
        chunk_key=K["chunk_a"],
        document_key=K["document_a"],
        chunk_number=2,
        confidence=0.75,
        artifact_key=K["document_artifact_a"],
        source_document_key=K["document_a"],
        head_mention_key=K["head_mention_a"],
        tail_mention_key=K["tail_mention_a"],
        relation_type="works_at",
        head_mapping_key=K["head_mapping_a"],
        tail_mapping_key=K["tail_mapping_a"],
        orientation=ProjectedEvidenceOrientationV1.HEAD_TO_TAIL,
        ontology_checksum=K["collection_ontology_checksum"],
        assembly_config_checksum=K["collection_assembly_checksum"],
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
        ProjectedAutomaticMembershipAuditV1(0, K["target_entity"], K["target_identity"], K["target_decision"], "resolver-v1"),
        ProjectedAutomaticMembershipAuditV1(0, K["source_entity"], K["source_identity"], K["source_decision"], "resolver-v1"),
    )
    audit_rows: tuple[object, ...] = memberships
    groups: tuple[ProjectedRelationGroupV1, ...] = ()
    if graph:
        physical = ProjectedPhysicalRelationAuditV1(
            1, K["relation_a"], K["collection_artifact_a"], K["source_entity"], "works_at", K["target_entity"]
        )
        relation_evidence = ProjectedRelationEvidenceAuditV1(1, _signature())
        groups = (
            ProjectedRelationGroupV1(
                K["source_identity"],
                "works_at",
                K["target_identity"],
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
            K["algorithm_signature"],
        ),
        caps=ProjectedSnapshotCapsV1(64, 10_000, 128, 2, 200, 1_000, 3_000, 3, 2),
        load_max_hops=2,
        allowed_scope=ProjectedAllowedScopeV1((K["document_a"],), (K["collection_a"],), K["scope_signature"]),
        identity_keys=(K["target_identity"], K["source_identity"]),
        seed_identities=(ProjectedSeedIdentityV1(K["seed_chunk_a"], K["source_identity"]),),
        relation_groups=groups,
        mentions=(),
        artifact_provenance=(
            _provenance(ProjectedScopeTypeV1.COLLECTION),
            _provenance(ProjectedScopeTypeV1.DOCUMENT),
        ),
        audit_rows=audit_rows,
    )


def test_exact_fields_and_closed_audit_tags_are_frozen() -> None:
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
    snapshot = _snapshot()
    assert tuple(row.kind for row in snapshot.audit_rows) == ("automatic_membership", "automatic_membership", "physical_relation", "relation_evidence")
    with pytest.raises(FrozenInstanceError):
        snapshot.load_max_hops = 1
    assert not hasattr(snapshot, "__dict__")


def test_strict_opaque_types_numbers_enums_and_subclasses_are_rejected() -> None:
    class Text(str): pass
    for value in (17, K["evidence_a"].upper(), Text(K["evidence_a"]), "f" * 63):
        with pytest.raises((TypeError, ValueError)): replace(_signature(), evidence_key=value)
    with pytest.raises(TypeError): replace(_signature(), relation_type=Text("works_at"))
    for value in (True, -1, 2**31):
        with pytest.raises((TypeError, ValueError)): replace(_signature(), chunk_number=value)
    for value in (1, float("nan"), float("inf"), -0.1, 1.1):
        with pytest.raises((TypeError, ValueError)): replace(_signature(), confidence=value)
    replace(_snapshot().relation_groups[0], raw_weight=2.0)
    for value in (2, True, 2.0000000001, float.fromhex("0x1.fffffffffffffp+1023")):
        with pytest.raises((TypeError, ValueError)): replace(_snapshot().relation_groups[0], raw_weight=value)
    with pytest.raises((TypeError, ValueError)): replace(_signature(), orientation="head_to_tail")
    with pytest.raises((TypeError, ValueError)): replace(_snapshot().caps, max_nodes=True)


def test_order_caps_and_all_snapshot_closures_are_enforced() -> None:
    snapshot = _snapshot()
    mutations = (
        {"identity_keys": tuple(reversed(snapshot.identity_keys))},
        {"identity_keys": (K["target_identity"], K["target_identity"])},
        {"seed_identities": (ProjectedSeedIdentityV1(K["seed_chunk_a"], K["other_identity"]),)},
        {"relation_groups": snapshot.relation_groups * 2},
        {"audit_rows": tuple(reversed(snapshot.audit_rows))},
        {"artifact_provenance": tuple(reversed(snapshot.artifact_provenance))},
        {"allowed_scope": ProjectedAllowedScopeV1((K["duplicate_document"],), (K["collection_a"],), K["scope_signature"])},
        {"audit_rows": snapshot.audit_rows[:2] + snapshot.audit_rows[3:]},
        {"audit_rows": snapshot.audit_rows[:1] + snapshot.audit_rows[2:]},
        {"caps": replace(snapshot.caps, max_nodes=1)},
        {"load_max_hops": 0},
    )
    for mutation in mutations:
        with pytest.raises((TypeError, ValueError)): replace(snapshot, **mutation)
    mention = ProjectedIdentityMentionV1(K["source_identity"], _chunk())
    with pytest.raises(ValueError, match="fallback"): replace(snapshot, mentions=(mention,))


def test_reviewed_audit_caps_and_provenance_roles_are_closed() -> None:
    base = _snapshot(graph=False); collapsed = replace(base.audit_rows[1], automatic_membership_key=K["target_identity"])
    with pytest.raises(ValueError, match="automatic|audit"): replace(base, caps=replace(base.caps, max_nodes=1), identity_keys=(K["target_identity"],), audit_rows=(base.audit_rows[0], collapsed))
    snapshot = _snapshot(); physical = snapshot.audit_rows[2]; evidence = snapshot.audit_rows[3]
    signature2 = replace(_signature(), evidence_key=K["evidence_b"], relation_key=K["relation_b"], relation_mention_key=K["relation_mention_b"], chunk_key=K["chunk_b"]); evidence2 = ProjectedRelationEvidenceAuditV1(1, signature2)
    physical2 = replace(physical, relation_key=K["relation_b"]); chunk2 = replace(_chunk(), chunk_key=K["chunk_b"], provenance_key=K["evidence_b"]); group = replace(snapshot.relation_groups[0], evidence=(chunk2, _chunk()))
    audits = (*snapshot.audit_rows[:2], physical, physical2, evidence2, evidence)
    with pytest.raises(ValueError, match="physical|audit"): replace(snapshot, caps=replace(snapshot.caps, max_edges=1), relation_groups=(group,), audit_rows=audits)
    with pytest.raises(ValueError, match="semantic|physical"): replace(snapshot, relation_groups=(group,), audit_rows=audits)
    physical2 = replace(physical2, relation_type="reports_to"); evidence2 = replace(evidence2, signature=replace(signature2, relation_type="reports_to")); audits = (*snapshot.audit_rows[:2], physical, physical2, evidence2, evidence)
    reverse2 = replace(snapshot.relation_groups[0], source_identity_key=K["target_identity"], relation_type="reports_to", target_identity_key=K["source_identity"], direction=ProjectedRetrievalDirectionV1.REVERSE_DIRECTED, evidence=(chunk2,))
    with pytest.raises(ValueError, match="evidence|audit"): replace(snapshot, caps=replace(snapshot.caps, max_evidence_rows=1, max_evidence_per_edge=1), relation_groups=(reverse2, replace(snapshot.relation_groups[0], evidence=(_chunk(),))), audit_rows=audits)
    mutations = (replace(physical, artifact_key=K["document_artifact_a"]), replace(evidence, signature=replace(_signature(), artifact_key=K["collection_artifact_a"])), replace(evidence, signature=replace(_signature(), relation_type="reports_to")), replace(evidence, signature=replace(_signature(), ontology_checksum=K["document_ontology_checksum"])), replace(evidence, signature=replace(_signature(), assembly_config_checksum=K["document_assembly_checksum"])))
    for mutation in mutations:
        with pytest.raises(ValueError): replace(snapshot, audit_rows=(*snapshot.audit_rows[:2], mutation if type(mutation) is ProjectedPhysicalRelationAuditV1 else physical, mutation) if type(mutation) is ProjectedRelationEvidenceAuditV1 else (*snapshot.audit_rows[:2], mutation, evidence))
    replace(snapshot, artifact_provenance=(snapshot.artifact_provenance[0], snapshot.artifact_provenance[1]))
def test_reviewed_strict_values_public_types_and_runtime_finality() -> None:
    class Text(str): pass
    for value in (None, False, Text("")):
        with pytest.raises((TypeError, ValueError)): replace(_provenance(ProjectedScopeTypeV1.DOCUMENT), embedding_model_signature=value)
    for value in ("", None, False, Text("embed-v1")):
        with pytest.raises((TypeError, ValueError)): replace(_provenance(ProjectedScopeTypeV1.COLLECTION), embedding_model_signature=value)
    for value in ("WorksAt", "works-at", "a" * 129):
        for row in (_signature(), _snapshot().relation_groups[0], _snapshot().audit_rows[2]):
            with pytest.raises(ValueError, match="relation_type"): replace(row, relation_type=value)
    for name in ("algorithm_version", "transition_version", "evidence_version", "seed_version"):
        with pytest.raises(ValueError, match=name): replace(_snapshot().algorithm, **{name: "wrong-v1"})
    assert get_type_hints(ProjectedAuthorizedGraphSnapshotV1)["audit_rows"] == tuple[ProjectedAuditRowV1, ...]
    tags = ((ProjectedAutomaticMembershipAuditV1, "automatic_membership"), (ProjectedPhysicalRelationAuditV1, "physical_relation"), (ProjectedRelationEvidenceAuditV1, "relation_evidence"), (ProjectedFallbackMentionAuditV1, "fallback_mention"))
    for kind, tag in tags: assert get_type_hints(kind)["kind"] == Literal[tag]
    with pytest.raises(TypeError, match="final"):
        class InvalidProjectedSubclass(ProjectedChunkEvidenceV1): pass
def test_evidence_reference_directions_and_fallback_semantics_are_bounded() -> None:
    snapshot = _snapshot(); forward = snapshot.relation_groups[0]; undirected = replace(forward, direction=ProjectedRetrievalDirectionV1.UNDIRECTED); reverse = replace(forward, source_identity_key=K["target_identity"], target_identity_key=K["source_identity"], direction=ProjectedRetrievalDirectionV1.REVERSE_DIRECTED)
    with pytest.raises(ValueError, match="evidence.*reference|direction"): replace(snapshot, relation_groups=(reverse, forward, undirected))
    with pytest.raises(ValueError, match="direction"): replace(snapshot, relation_groups=(forward, undirected))
    replace(snapshot, relation_groups=(reverse, forward))
    replace(snapshot, relation_groups=(replace(undirected, source_identity_key=K["target_identity"], target_identity_key=K["source_identity"]), undirected))
    base = _snapshot(graph=False); first = replace(_chunk(), confidence=0.5); second = _chunk(); fallbacks = (ProjectedFallbackMentionAuditV1(0, K["source_identity"], first), ProjectedFallbackMentionAuditV1(0, K["source_identity"], second))
    with pytest.raises(ValueError, match="fallback.*duplicate|semantic"): replace(base, mentions=(ProjectedIdentityMentionV1(K["source_identity"], second),), audit_rows=(*base.audit_rows, *fallbacks))
    conflict = replace(second, provenance_key=K["conflicting_provenance"], chunk_number=3); conflict_rows = (ProjectedFallbackMentionAuditV1(0, K["source_identity"], second), ProjectedFallbackMentionAuditV1(0, K["source_identity"], conflict))
    with pytest.raises(ValueError, match="coordinate"): replace(base, mentions=(ProjectedIdentityMentionV1(K["source_identity"], conflict), ProjectedIdentityMentionV1(K["source_identity"], second)), audit_rows=(*base.audit_rows, *conflict_rows))
def test_scope_cardinality_discovery_hops_and_seed_caps_are_complete() -> None:
    snapshot = _snapshot(); collection, document = snapshot.artifact_provenance; duplicate_collection = replace(collection, artifact_key=K["duplicate_artifact"], build_key=K["duplicate_build"]); duplicate_document = replace(document, artifact_key=K["duplicate_artifact"], build_key=K["duplicate_build"])
    for rows in ((duplicate_collection, collection, document), (collection, document, duplicate_document)):
        with pytest.raises(ValueError, match="scope|provenance"): replace(snapshot, artifact_provenance=rows)
    collection2 = replace(collection, artifact_key=K["duplicate_artifact"], scope_key=K["duplicate_collection"], collection_key=K["duplicate_collection"], build_key=K["duplicate_build"]); document1 = replace(document, collection_key=K["duplicate_collection"]); document2 = replace(document, artifact_key=K["evidence_a"], scope_key=K["duplicate_document"], collection_key=K["duplicate_collection"], build_key=K["evidence_b"])
    with pytest.raises(ValueError, match="collection|provenance"): replace(snapshot, allowed_scope=ProjectedAllowedScopeV1((K["document_a"], K["duplicate_document"]), (K["duplicate_collection"], K["collection_a"]), K["scope_signature"]), artifact_provenance=(collection2, collection, document1, document2))
    with pytest.raises(ValueError, match="hop"): replace(snapshot, audit_rows=(*snapshot.audit_rows[:3], replace(snapshot.audit_rows[3], discovery_hop=2)))
    raw = _snapshot(graph=False); collapsed = replace(raw.audit_rows[1], automatic_membership_key=K["target_identity"], discovery_hop=1); base = replace(raw, identity_keys=(K["target_identity"],), seed_identities=(ProjectedSeedIdentityV1(K["seed_chunk_a"], K["target_identity"]),), audit_rows=(raw.audit_rows[0], collapsed)); fallback = ProjectedFallbackMentionAuditV1(1, K["target_identity"], _chunk())
    with pytest.raises(ValueError, match="hop"): replace(base, mentions=(ProjectedIdentityMentionV1(K["target_identity"], _chunk()),), audit_rows=(*base.audit_rows, fallback))
    replace(snapshot, caps=replace(snapshot.caps, max_seeds=1), seed_identities=(ProjectedSeedIdentityV1(K["seed_chunk_a"], K["target_identity"]), ProjectedSeedIdentityV1(K["seed_chunk_a"], K["source_identity"])))
    with pytest.raises(ValueError, match="seed"): replace(snapshot, caps=replace(snapshot.caps, max_seeds=1), seed_identities=(ProjectedSeedIdentityV1(K["seed_chunk_b"], K["target_identity"]), ProjectedSeedIdentityV1(K["seed_chunk_a"], K["source_identity"])))
    with pytest.raises(ValueError, match="seed|cap"): replace(snapshot, caps=replace(snapshot.caps, max_seeds=2, max_nodes=2), seed_identities=(ProjectedSeedIdentityV1(K["seed_chunk_b"], K["target_identity"]), ProjectedSeedIdentityV1(K["seed_chunk_a"], K["target_identity"]), ProjectedSeedIdentityV1(K["seed_chunk_a"], K["source_identity"])))
def test_physical_and_group_hops_bind_to_identity_discovery() -> None:
    snapshot = _snapshot(); source = snapshot.audit_rows[1]; target = replace(snapshot.audit_rows[0], discovery_hop=1); physical = snapshot.audit_rows[2]; evidence = snapshot.audit_rows[3]
    reverse = replace(snapshot.relation_groups[0], source_identity_key=K["target_identity"], target_identity_key=K["source_identity"], direction=ProjectedRetrievalDirectionV1.REVERSE_DIRECTED, admission_hop=2); mixed = replace(snapshot, relation_groups=(reverse,), audit_rows=(source, target, physical, evidence))
    bad_physical = replace(physical, discovery_hop=2); bad_evidence = replace(evidence, discovery_hop=2)
    with pytest.raises(ValueError, match="physical.*hop|discovery"): replace(mixed, audit_rows=(source, target, bad_physical, bad_evidence))
    with pytest.raises(ValueError, match="group.*hop|admission"): replace(mixed, relation_groups=(replace(reverse, admission_hop=1),))
def test_rows_reject_over_cap_before_touching_elements() -> None:
    touched = 0
    def touch(row):
        nonlocal touched; touched += 1; return row
    with pytest.raises(ValueError, match="cap"): projected_types_module._rows((object(),), object, "hostile", 0, touch)
    assert touched == 0
def test_public_stub_matches_runtime_dataclass_contract() -> None:
    tree = ast.parse(Path(projected_types_module.__file__).with_suffix(".pyi").read_text(encoding="utf-8")); classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}; functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    dto_names = ("ProjectedEvidenceSignatureV1", "ProjectedChunkEvidenceV1", "ProjectedSeedIdentityV1", "ProjectedRelationGroupV1", "ProjectedIdentityMentionV1", "ProjectedArtifactProvenanceV1", "ProjectedAllowedScopeV1", "ProjectedAlgorithmSignatureV1", "ProjectedSnapshotCapsV1", "ProjectedAutomaticMembershipAuditV1", "ProjectedPhysicalRelationAuditV1", "ProjectedRelationEvidenceAuditV1", "ProjectedFallbackMentionAuditV1", "ProjectedAuthorizedGraphSnapshotV1")
    enum_names = ("ProjectedEvidenceOrientationV1", "ProjectedRetrievalDirectionV1", "ProjectedScopeTypeV1"); expected_public = {*dto_names, *enum_names, "ProjectedAuditRowV1", "canonical_projected_snapshot_bytes", "projected_snapshot_checksum"}
    aliases = {node.name.id for node in tree.body if isinstance(node, ast.TypeAlias) and isinstance(node.name, ast.Name)}; assert set(classes) | set(functions) | aliases == expected_public
    for name in dto_names:
        node = classes[name]; annotations = [item for item in node.body if isinstance(item, ast.AnnAssign)]; assert tuple(item.target.id for item in annotations) == tuple(item.name for item in fields(getattr(projected_types_module, name)))
        decorators = {ast.unparse(item) for item in node.decorator_list}; assert "final" in decorators and "dataclass(frozen=True, slots=True)" in decorators
    for name in enum_names: assert "final" in {ast.unparse(item) for item in classes[name].decorator_list}
    tag_types = {"ProjectedAutomaticMembershipAuditV1": "Literal['automatic_membership']", "ProjectedPhysicalRelationAuditV1": "Literal['physical_relation']", "ProjectedRelationEvidenceAuditV1": "Literal['relation_evidence']", "ProjectedFallbackMentionAuditV1": "Literal['fallback_mention']"}
    for name, expected in tag_types.items(): assert ast.unparse(next(item.annotation for item in classes[name].body if isinstance(item, ast.AnnAssign) and item.target.id == "kind")) == expected
    assert ast.unparse(next(item.annotation for item in classes["ProjectedAuthorizedGraphSnapshotV1"].body if isinstance(item, ast.AnnAssign) and item.target.id == "audit_rows")) == "tuple[ProjectedAuditRowV1, ...]"
    for name, result in (("canonical_projected_snapshot_bytes", "bytes"), ("projected_snapshot_checksum", "str")): assert ast.unparse(functions[name].args.args[0].annotation) == "ProjectedAuthorizedGraphSnapshotV1" and ast.unparse(functions[name].returns) == result
def test_canonical_snapshot_vector_and_private_integer_canary_exclusion() -> None:
    snapshot = _snapshot(graph=False)
    encoded = canonical_projected_snapshot_bytes(snapshot)
    assert encoded == EXPECTED_SNAPSHOT_BYTES
    assert b'"confidence":"0x1.8000000000000p-1"' in canonical_projected_snapshot_bytes(_snapshot())
    assert projected_snapshot_checksum(snapshot) == sha256(encoded).hexdigest()
    assert projected_snapshot_checksum(snapshot) == "d0a514aae7348b66d4d3a0fbbfd6a17e1d6a8b9dedfece6dfb9645d3a7c157a0"
    private = PrivateProjectionChunkReferenceV1(K["chunk_a"], 9_223_372_036_854_775_807, "12345678-1234-5678-9234-567812345678", 2)
    with pytest.raises(TypeError): replace(snapshot, mentions=(private,))
    with pytest.raises(TypeError): canonical_projected_snapshot_bytes(private)
    assert b"9223372036854775807" not in encoded
