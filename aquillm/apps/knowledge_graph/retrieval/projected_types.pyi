"""Static public contract for provider-neutral projected ranking snapshots."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal, final

@final
class ProjectedEvidenceOrientationV1(StrEnum):
    HEAD_TO_TAIL: Literal["head_to_tail"]
    TAIL_TO_HEAD: Literal["tail_to_head"]

@final
class ProjectedRetrievalDirectionV1(StrEnum):
    FORWARD: Literal["forward"]
    REVERSE_DIRECTED: Literal["reverse_directed"]
    UNDIRECTED: Literal["undirected"]

@final
class ProjectedScopeTypeV1(StrEnum):
    COLLECTION: Literal["collection"]
    DOCUMENT: Literal["document"]

@final
@dataclass(frozen=True, slots=True)
class ProjectedEvidenceSignatureV1:
    evidence_key: str
    relation_key: str
    relation_mention_key: str
    chunk_key: str
    document_key: str
    chunk_number: int
    confidence: float
    artifact_key: str
    source_document_key: str
    head_mention_key: str
    tail_mention_key: str
    relation_type: str
    head_mapping_key: str
    tail_mapping_key: str
    orientation: ProjectedEvidenceOrientationV1
    ontology_checksum: str
    assembly_config_checksum: str

@final
@dataclass(frozen=True, slots=True)
class ProjectedChunkEvidenceV1:
    chunk_key: str
    document_key: str
    chunk_number: int
    confidence: float
    provenance_key: str

@final
@dataclass(frozen=True, slots=True)
class ProjectedSeedIdentityV1:
    seed_chunk_key: str
    identity_key: str

@final
@dataclass(frozen=True, slots=True)
class ProjectedRelationGroupV1:
    source_identity_key: str
    relation_type: str
    target_identity_key: str
    direction: ProjectedRetrievalDirectionV1
    raw_weight: float
    admission_hop: int
    evidence: tuple[ProjectedChunkEvidenceV1, ...]

@final
@dataclass(frozen=True, slots=True)
class ProjectedIdentityMentionV1:
    identity_key: str
    evidence: ProjectedChunkEvidenceV1

@final
@dataclass(frozen=True, slots=True)
class ProjectedArtifactProvenanceV1:
    artifact_key: str
    scope_type: ProjectedScopeTypeV1
    scope_key: str
    collection_key: str
    rebuild_request_key: str | None
    evaluation_only: bool
    build_key: str
    build_generation: int
    orchestration_version: int
    source_hash: str
    ontology_version: str
    ontology_checksum: str
    extractor_version: str
    resolver_version: str
    resolution_config_checksum: str
    filter_policy_version: str
    filter_policy_checksum: str
    embedding_model_signature: str
    assembly_version: str
    assembly_config_checksum: str

@final
@dataclass(frozen=True, slots=True)
class ProjectedAllowedScopeV1:
    document_keys: tuple[str, ...]
    collection_keys: tuple[str, ...]
    scope_version_signature: str

@final
@dataclass(frozen=True, slots=True)
class ProjectedAlgorithmSignatureV1:
    algorithm_version: str
    transition_version: str
    evidence_version: str
    seed_version: str
    algorithm_signature: str

@final
@dataclass(frozen=True, slots=True)
class ProjectedSnapshotCapsV1:
    max_seeds: int
    max_scope_documents: int
    max_scope_collections: int
    max_hops: int
    max_nodes: int
    max_edges: int
    max_evidence_rows: int
    max_evidence_per_edge: int
    max_mentions_per_entity: int

@final
@dataclass(frozen=True, slots=True)
class ProjectedAutomaticMembershipAuditV1:
    discovery_hop: int
    entity_key: str
    automatic_membership_key: str
    decision_checksum: str
    resolver_version: str
    kind: Literal["automatic_membership"] = field(
        init=False, default="automatic_membership"
    )

@final
@dataclass(frozen=True, slots=True)
class ProjectedPhysicalRelationAuditV1:
    discovery_hop: int
    relation_key: str
    artifact_key: str
    source_entity_key: str
    relation_type: str
    target_entity_key: str
    kind: Literal["physical_relation"] = field(init=False, default="physical_relation")

@final
@dataclass(frozen=True, slots=True)
class ProjectedRelationEvidenceAuditV1:
    discovery_hop: int
    signature: ProjectedEvidenceSignatureV1
    kind: Literal["relation_evidence"] = field(init=False, default="relation_evidence")

@final
@dataclass(frozen=True, slots=True)
class ProjectedFallbackMentionAuditV1:
    discovery_hop: int
    identity_key: str
    evidence: ProjectedChunkEvidenceV1
    kind: Literal["fallback_mention"] = field(init=False, default="fallback_mention")

type ProjectedAuditRowV1 = (
    ProjectedAutomaticMembershipAuditV1
    | ProjectedFallbackMentionAuditV1
    | ProjectedPhysicalRelationAuditV1
    | ProjectedRelationEvidenceAuditV1
)

@final
@dataclass(frozen=True, slots=True)
class ProjectedAuthorizedGraphSnapshotV1:
    algorithm: ProjectedAlgorithmSignatureV1
    caps: ProjectedSnapshotCapsV1
    load_max_hops: int
    allowed_scope: ProjectedAllowedScopeV1
    identity_keys: tuple[str, ...]
    seed_identities: tuple[ProjectedSeedIdentityV1, ...]
    relation_groups: tuple[ProjectedRelationGroupV1, ...]
    mentions: tuple[ProjectedIdentityMentionV1, ...]
    artifact_provenance: tuple[ProjectedArtifactProvenanceV1, ...]
    audit_rows: tuple[ProjectedAuditRowV1, ...]

def canonical_projected_snapshot_bytes(
    snapshot: ProjectedAuthorizedGraphSnapshotV1,
) -> bytes: ...
def projected_snapshot_checksum(
    snapshot: ProjectedAuthorizedGraphSnapshotV1,
) -> str: ...
