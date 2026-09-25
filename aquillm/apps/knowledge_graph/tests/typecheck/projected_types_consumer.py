"""Positive Pyright consumer for the projected snapshot enum contract."""

from apps.knowledge_graph.retrieval.projected_types import (
    ProjectedArtifactProvenanceV1,
    ProjectedChunkEvidenceV1,
    ProjectedEvidenceOrientationV1,
    ProjectedEvidenceSignatureV1,
    ProjectedRelationGroupV1,
    ProjectedRetrievalDirectionV1,
    ProjectedScopeTypeV1,
)

_KEY = "0" * 64

_orientation: ProjectedEvidenceOrientationV1 = (
    ProjectedEvidenceOrientationV1.HEAD_TO_TAIL
)
_signature: ProjectedEvidenceSignatureV1 = ProjectedEvidenceSignatureV1(
    evidence_key=_KEY,
    relation_key=_KEY,
    relation_mention_key=_KEY,
    chunk_key=_KEY,
    document_key=_KEY,
    chunk_number=0,
    confidence=1.0,
    artifact_key=_KEY,
    source_document_key=_KEY,
    head_mention_key=_KEY,
    tail_mention_key=_KEY,
    relation_type="works_at",
    head_mapping_key=_KEY,
    tail_mapping_key=_KEY,
    orientation=ProjectedEvidenceOrientationV1.HEAD_TO_TAIL,
    ontology_checksum=_KEY,
    assembly_config_checksum=_KEY,
)
_chunk = ProjectedChunkEvidenceV1(_KEY, _KEY, 0, 1.0, _KEY)
_direction: ProjectedRetrievalDirectionV1 = ProjectedRetrievalDirectionV1.FORWARD
_group: ProjectedRelationGroupV1 = ProjectedRelationGroupV1(
    source_identity_key=_KEY,
    relation_type="works_at",
    target_identity_key=_KEY,
    direction=ProjectedRetrievalDirectionV1.FORWARD,
    raw_weight=1.0,
    admission_hop=1,
    evidence=(_chunk,),
)
_scope_type: ProjectedScopeTypeV1 = ProjectedScopeTypeV1.COLLECTION
_provenance: ProjectedArtifactProvenanceV1 = ProjectedArtifactProvenanceV1(
    artifact_key=_KEY,
    scope_type=ProjectedScopeTypeV1.COLLECTION,
    scope_key=_KEY,
    collection_key=_KEY,
    rebuild_request_key=None,
    evaluation_only=False,
    build_key=_KEY,
    build_generation=1,
    orchestration_version=1,
    source_hash=_KEY,
    ontology_version="ontology-v1",
    ontology_checksum=_KEY,
    extractor_version="extractor-v1",
    resolver_version="resolver-v1",
    resolution_config_checksum=_KEY,
    filter_policy_version="filter-v1",
    filter_policy_checksum=_KEY,
    embedding_model_signature="embed-v1",
    assembly_version="assembly-v1",
    assembly_config_checksum=_KEY,
)
