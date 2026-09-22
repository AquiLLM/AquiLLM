"""Immutable relation and evidence plan values for collection assembly."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class EvidenceDisposition(StrEnum):
    PROMOTED = "promoted"
    SUPPRESSED = "suppressed"
    REJECTED = "rejected"
RelationKey = tuple[int, str, int]


@dataclass(frozen=True, slots=True)
class PlannedRelation:
    source_entity_id: int
    relation_type: str
    target_entity_id: int
    support_count: int
    confidence: float
    evidence_mention_ids: tuple[int, ...]

    @property
    def key(self) -> RelationKey:
        return self.source_entity_id, self.relation_type, self.target_entity_id


@dataclass(frozen=True, slots=True)
class PlannedEvidence:
    relation_mention_id: int
    disposition: EvidenceDisposition
    reason: str
    relation_key: RelationKey | None
    head_mapping_id: int | None
    tail_mapping_id: int | None
    orientation: str


@dataclass(frozen=True, slots=True)
class CollectionAssemblyPlan:
    relations: tuple[PlannedRelation, ...]
    evidence: tuple[PlannedEvidence, ...]
    checksum: str


@dataclass(frozen=True, slots=True)
class AssemblyProjectionStats:
    entity_count: int
    relation_count: int
    evidence_count: int
    promoted_evidence_count: int
    suppressed_evidence_count: int
    rejected_evidence_count: int
    orphan_count: int
    orphan_ratio: float
