from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from .serialization import (
    _count,
    _key,
    _token,
    _uuid,
    _validate_bundle,
    _ValidatedRecord,
)

_MAX_PRIVATE_PK = 2**63 - 1
_MAX_ATTEMPTS = 32767


class ProjectionLifecycleState(StrEnum):
    PENDING = "pending"
    BUILDING = "building"
    READY = "ready"
    FAILED = "failed"
    SUPERSEDED = "superseded"


class ProjectionFailureCode(StrEnum):
    SOURCE_CHANGED = "source_changed"
    LEASE_LOST = "lease_lost"
    GRAPH_UNAVAILABLE = "graph_unavailable"
    WRITE_FAILED = "write_failed"
    VALIDATION_FAILED = "validation_failed"
    CHECKSUM_MISMATCH = "checksum_mismatch"
    TIMEOUT = "timeout"
    INTERNAL_ERROR = "internal_error"


@dataclass(frozen=True, slots=True)
class ProjectionGenerationMarkerV1(_ValidatedRecord):
    generation_key: str
    collection_key: str
    artifact_key: str
    schema_version: str
    projection_version: str
    identifier_key_version: str
    membership_epoch: int
    membership_checksum: str

    _key_fields = (
        "generation_key collection_key artifact_key membership_checksum".split()
    )
    _token_fields = "schema_version projection_version identifier_key_version".split()
    _count_fields = ("membership_epoch",)


@dataclass(frozen=True, slots=True)
class ProjectedEntityV1(_ValidatedRecord):
    entity_key: str
    generation_key: str
    artifact_key: str
    collection_key: str
    ontology_type: str
    cluster_key: str
    retrieval_utility: float

    _key_fields = (
        "entity_key generation_key artifact_key collection_key cluster_key"
    ).split()
    _token_fields = ("ontology_type",)
    _float_fields = ("retrieval_utility",)


@dataclass(frozen=True, slots=True)
class AutomaticCanonicalMembershipV1(_ValidatedRecord):
    entity_key: str
    automatic_membership_key: str | None
    decision_checksum: str
    resolver_version: str
    resolution_config_checksum: str

    _key_fields = "entity_key decision_checksum resolution_config_checksum".split()
    _token_fields = ("resolver_version",)

    def __post_init__(self) -> None:
        _ValidatedRecord.__post_init__(self)
        if self.automatic_membership_key is not None:
            _key(self.automatic_membership_key, "automatic_membership_key")


@dataclass(frozen=True, slots=True)
class ProjectedDocumentMembershipV1(_ValidatedRecord):
    document_key: str
    generation_key: str

    _key_fields = "document_key generation_key".split()


@dataclass(frozen=True, slots=True)
class ProjectedChunkMembershipV1(_ValidatedRecord):
    chunk_key: str
    document_key: str
    chunk_number: int

    _key_fields = "chunk_key document_key".split()
    _count_fields = ("chunk_number",)


@dataclass(frozen=True, slots=True)
class ProjectedPhysicalRelationV1(_ValidatedRecord):
    relation_key: str
    artifact_key: str
    source_entity_key: str
    relation_type: str
    target_entity_key: str

    _key_fields = (
        "relation_key artifact_key source_entity_key target_entity_key".split()
    )
    _token_fields = ("relation_type",)


@dataclass(frozen=True, slots=True)
class ProjectedRelationEvidenceV1(_ValidatedRecord):
    evidence_key: str
    relation_key: str
    relation_mention_key: str
    chunk_key: str
    document_key: str
    chunk_number: int
    confidence: float
    source_document_key: str
    head_mention_key: str
    tail_mention_key: str
    head_mapping_key: str
    tail_mapping_key: str
    orientation: str
    ontology_checksum: str
    assembly_config_checksum: str
    provenance_key: str
    semantic_signature: str

    _key_fields = (
        "evidence_key relation_key relation_mention_key chunk_key document_key "
        "source_document_key head_mention_key tail_mention_key head_mapping_key "
        "tail_mapping_key ontology_checksum assembly_config_checksum provenance_key "
        "semantic_signature"
    ).split()
    _token_fields = ("orientation",)
    _count_fields = ("chunk_number",)
    _float_fields = ("confidence",)


@dataclass(frozen=True, slots=True)
class ProjectedArtifactProvenanceV1(_ValidatedRecord):
    artifact_key: str
    scope_type: str
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

    _key_fields = (
        "artifact_key scope_key collection_key build_key source_hash "
        "ontology_checksum resolution_config_checksum filter_policy_checksum "
        "assembly_config_checksum"
    ).split()
    _token_fields = (
        "scope_type ontology_version extractor_version resolver_version "
        "filter_policy_version embedding_model_signature assembly_version"
    ).split()
    _count_fields = ("build_generation", "orchestration_version")

    def __post_init__(self) -> None:
        _ValidatedRecord.__post_init__(self)
        if self.rebuild_request_key is not None:
            _key(self.rebuild_request_key, "rebuild_request_key")
        if self.scope_type not in {"collection", "document"}:
            raise ValueError("scope_type is invalid")
        if type(self.evaluation_only) is not bool:
            raise TypeError("evaluation_only must be a bool")


@dataclass(frozen=True, slots=True)
class ProjectionCountsV1(_ValidatedRecord):
    entity_count: int
    automatic_membership_count: int
    document_count: int
    chunk_count: int
    relation_count: int
    evidence_count: int
    artifact_provenance_count: int

    _count_fields = (
        "entity_count automatic_membership_count document_count chunk_count "
        "relation_count evidence_count artifact_provenance_count"
    ).split()


@dataclass(frozen=True, slots=True)
class ProjectionGenerationManifestV1(_ValidatedRecord):
    generation_key: str
    schema_version: str
    projection_version: str
    identifier_key_version: str
    graph_checksum: str
    snapshot_checksum: str
    private_mapping_checksum: str
    counts: ProjectionCountsV1
    state: ProjectionLifecycleState

    _key_fields = (
        "generation_key graph_checksum snapshot_checksum private_mapping_checksum"
    ).split()
    _token_fields = "schema_version projection_version identifier_key_version".split()

    def __post_init__(self) -> None:
        _ValidatedRecord.__post_init__(self)
        if type(self.counts) is not ProjectionCountsV1:
            raise TypeError("counts must be ProjectionCountsV1")
        if type(self.state) is not ProjectionLifecycleState:
            raise TypeError("state must be ProjectionLifecycleState")


@dataclass(frozen=True, slots=True)
class ProjectionLeaseV1:
    projection_id: str
    owner: str
    expires_at: datetime
    attempt_count: int

    def __post_init__(self) -> None:
        _uuid(self.projection_id, "projection_id")
        _token(self.owner, "owner", maximum=128)
        if type(self.expires_at) is not datetime or self.expires_at.tzinfo is not UTC:
            raise ValueError("expires_at must be an exact UTC datetime")
        _count(self.attempt_count, "attempt_count", maximum=_MAX_ATTEMPTS)


@dataclass(frozen=True, slots=True)
class ProjectionFailureStateV1:
    state: ProjectionLifecycleState
    failure_code: ProjectionFailureCode
    attempt_count: int

    def __post_init__(self) -> None:
        if self.state is not ProjectionLifecycleState.FAILED:
            raise ValueError("failure state must be failed")
        if type(self.failure_code) is not ProjectionFailureCode:
            raise TypeError("failure_code must be ProjectionFailureCode")
        _count(self.attempt_count, "attempt_count", maximum=_MAX_ATTEMPTS)


@dataclass(frozen=True, slots=True)
class PrivateProjectionChunkReferenceV1:
    projection_chunk_key: str
    integer_chunk_pk: int
    document_uuid: str
    chunk_number: int

    def __post_init__(self) -> None:
        _key(self.projection_chunk_key, "projection_chunk_key")
        _count(
            self.integer_chunk_pk,
            "integer_chunk_pk",
            minimum=1,
            maximum=_MAX_PRIVATE_PK,
        )
        _uuid(self.document_uuid, "document_uuid")
        _count(self.chunk_number, "chunk_number")


@dataclass(frozen=True, slots=True)
class CollectionGraphProjectionBundleV1:
    generation: ProjectionGenerationMarkerV1
    entities: tuple[ProjectedEntityV1, ...]
    automatic_memberships: tuple[AutomaticCanonicalMembershipV1, ...]
    documents: tuple[ProjectedDocumentMembershipV1, ...]
    chunks: tuple[ProjectedChunkMembershipV1, ...]
    relations: tuple[ProjectedPhysicalRelationV1, ...]
    evidence: tuple[ProjectedRelationEvidenceV1, ...]
    artifact_provenance: tuple[ProjectedArtifactProvenanceV1, ...]
    counts: ProjectionCountsV1

    def __post_init__(self) -> None:
        _validate_bundle(self)
