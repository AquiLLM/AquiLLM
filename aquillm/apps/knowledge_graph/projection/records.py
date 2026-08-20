from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import UTC, datetime
from enum import StrEnum
from math import isfinite

from .serialization import (
    _count,
    _key,
    _token,
    _uuid,
    _validate_bundle,
)

_MAX_PRIVATE_PK = 2**63 - 1
_MAX_ATTEMPTS = 32767


def _finite_float(value: object, name: str) -> None:
    if type(value) is not float:
        raise TypeError(f"{name} must be a built-in float")
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")


class _ValidatedRecord:
    __slots__ = ()

    def __post_init__(self) -> None:
        optional = {"automatic_membership_key", "rebuild_request_key"}
        for field in fields(self):
            value = getattr(self, field.name)
            if value is None:
                if field.name not in optional:
                    raise TypeError(f"{field.name} must not be null")
                continue
            if field.name.endswith(("_key", "_checksum")) or field.name in {
                "source_hash",
                "semantic_signature",
            }:
                _key(value, field.name)
            elif field.name in {"retrieval_utility", "confidence"}:
                _finite_float(value, field.name)
                if not 0.0 <= value <= 1.0:
                    raise ValueError(f"{field.name} must be in the unit interval")
            elif field.name == "embedding_model_signature":
                if type(value) is not str:
                    raise TypeError(f"{field.name} must be a built-in str")
                if value:
                    _token(value, field.name)
            elif type(value) is str:
                _token(value, field.name)
            elif type(value) is int:
                _count(value, field.name)
            elif type(value) is float:
                _finite_float(value, field.name)
            elif (type(value) is bool and field.name == "evaluation_only") or (
                field.name in {"counts", "state"}
            ):
                continue
            else:
                raise TypeError(f"{field.name} has an unsupported exact type")


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


@dataclass(frozen=True, slots=True)
class ProjectedEntityV1(_ValidatedRecord):
    entity_key: str
    generation_key: str
    artifact_key: str
    collection_key: str
    ontology_type: str
    cluster_key: str
    retrieval_utility: float


@dataclass(frozen=True, slots=True)
class AutomaticCanonicalMembershipV1(_ValidatedRecord):
    entity_key: str
    automatic_membership_key: str | None
    decision_checksum: str
    resolver_version: str
    resolution_config_checksum: str


@dataclass(frozen=True, slots=True)
class ProjectedDocumentMembershipV1(_ValidatedRecord):
    document_key: str
    generation_key: str


@dataclass(frozen=True, slots=True)
class ProjectedChunkMembershipV1(_ValidatedRecord):
    chunk_key: str
    document_key: str
    chunk_number: int


@dataclass(frozen=True, slots=True)
class ProjectedPhysicalRelationV1(_ValidatedRecord):
    relation_key: str
    artifact_key: str
    source_entity_key: str
    relation_type: str
    target_entity_key: str


@dataclass(frozen=True, slots=True)
class ProjectedRelationEvidenceV1(_ValidatedRecord):
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
    head_mapping_key: str
    tail_mapping_key: str
    orientation: str
    relation_type: str
    ontology_checksum: str
    assembly_config_checksum: str
    provenance_key: str
    semantic_signature: str

    def __post_init__(self) -> None:
        _ValidatedRecord.__post_init__(self)
        if self.orientation not in {"head_to_tail", "tail_to_head"}:
            raise ValueError("orientation is invalid")


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

    def __post_init__(self) -> None:
        _ValidatedRecord.__post_init__(self)
        if self.rebuild_request_key is not None:
            _key(self.rebuild_request_key, "rebuild_request_key")
        if self.scope_type not in {"collection", "document"}:
            raise ValueError("scope_type is invalid")
        if self.scope_type == "collection" and not self.embedding_model_signature:
            raise ValueError("collection embedding_model_signature must be nonempty")
        if self.scope_type == "document" and self.embedding_model_signature:
            raise ValueError("document embedding_model_signature must be empty")
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
