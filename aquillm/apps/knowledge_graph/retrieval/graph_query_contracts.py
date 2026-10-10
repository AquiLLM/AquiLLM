"""Versioned private relation requests and bounded evidence topology."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from ._graph_v2_validation import (
    checksum,
    ordered_strings,
    reference_keys,
    relation,
    score,
)
from .graph_evidence_associations_v2 import (
    GraphEvidenceAssociationV2,
    validate_associations,
)
from .graph_handoff import GraphCoverageV2
from .projected_types import ProjectedAuthorizedGraphSnapshotV1


@dataclass(frozen=True, slots=True)
class GraphRelationRequestV2:
    ontology_checksum: str
    predicate_weights: tuple[tuple[str, float], ...]
    policy_version: str
    query_digest: str

    def __post_init__(self):
        checksum(self.ontology_checksum, "ontology_checksum")
        checksum(self.query_digest, "query_digest")
        if (
            type(self.policy_version) is not str
            or self.policy_version != "relation-request-v2"
        ):
            raise ValueError("unsupported relation request policy")
        if (
            type(self.predicate_weights) is not tuple
            or len(self.predicate_weights) > 128
        ):
            raise TypeError("predicate_weights must be a bounded exact tuple")
        for row in self.predicate_weights:
            if type(row) is not tuple or len(row) != 2:
                raise TypeError("predicate weights must be exact pairs")
            relation(row[0])
            score(row[1], "predicate weight", maximum=1.0)
            if row[1] not in (0.2, 1.0):
                raise ValueError("unsupported predicate weight")
        names = tuple(row[0] for row in self.predicate_weights)
        if names != tuple(sorted(set(names))):
            raise ValueError("predicate weights must be unique and sorted")
        weights = tuple(row[1] for row in self.predicate_weights)
        if 0.2 in weights and weights.count(1.0) != 1:
            raise ValueError("weighted intent requires exactly one preferred predicate")


@dataclass(frozen=True, slots=True)
class GraphGroupRelationRequestV2:
    authority_group_checksum: str
    request: GraphRelationRequestV2

    def __post_init__(self):
        checksum(self.authority_group_checksum, "authority_group_checksum")
        if type(self.request) is not GraphRelationRequestV2:
            raise TypeError("request must be an exact GraphRelationRequestV2")


@dataclass(frozen=True, slots=True)
class GraphEvidenceRefV2:
    chunk_key: str
    graph_generation_key: str
    representation_generation_key: UUID | None
    authority_group_checksum: str
    identity_keys: tuple[str, ...]
    evidence_provenance_keys: tuple[str, ...]
    relation_types: tuple[str, ...]
    confidence: float
    associations: tuple[GraphEvidenceAssociationV2, ...]

    def __post_init__(self):
        reference_keys(self)
        ordered_strings(
            self.identity_keys,
            "identity_keys",
            lambda k: checksum(k, "identity key"),
            200,
            required=True,
        )
        score(self.confidence, "confidence", maximum=1.0)
        validate_associations(self)


@dataclass(frozen=True, slots=True)
class BoundedGraphTopologyV2:
    snapshot: ProjectedAuthorizedGraphSnapshotV1
    evidence_candidates: tuple[GraphEvidenceRefV2, ...]
    coverage: GraphCoverageV2
    relation_request_digest: str
    authority_group_checksum: str

    def __post_init__(self):
        if type(self.snapshot) is not ProjectedAuthorizedGraphSnapshotV1:
            raise TypeError("snapshot must be an exact projected snapshot")
        if type(self.coverage) is not GraphCoverageV2:
            raise TypeError("coverage must be an exact GraphCoverageV2")
        checksum(self.relation_request_digest, "relation_request_digest")
        checksum(self.authority_group_checksum, "authority_group_checksum")
        candidates = self.evidence_candidates
        if type(candidates) is not tuple or any(
            type(c) is not GraphEvidenceRefV2 for c in candidates
        ):
            raise TypeError(
                "evidence_candidates must contain exact GraphEvidenceRefV2 values"
            )
        if len(candidates) > self.snapshot.caps.max_evidence_rows:
            raise ValueError("evidence_candidates exceeds the evidence bound")
        if (
            sum(len(c.associations) for c in candidates)
            > self.snapshot.caps.max_evidence_rows
        ):
            raise ValueError("associations exceed the snapshot evidence bound")
        provenance = tuple(a.provenance_key for c in candidates for a in c.associations)
        if len(provenance) != len(set(provenance)):
            raise ValueError("association provenance is repeated across chunks")
        keys = tuple(c.chunk_key for c in candidates)
        if keys != tuple(sorted(set(keys))):
            raise ValueError("evidence_candidates must be unique and chunk-key sorted")
        identities = set(self.snapshot.identity_keys)
        for candidate in candidates:
            if candidate.authority_group_checksum != self.authority_group_checksum:
                raise ValueError("evidence belongs to a different authority group")
            if not set(candidate.identity_keys) <= identities:
                raise ValueError("evidence identities must belong to snapshot")
        if candidates and (
            self.coverage.status == "unavailable" or not self.coverage.ready_documents
        ):
            raise ValueError("unavailable graph cannot carry evidence candidates")


__all__ = [
    "GraphRelationRequestV2",
    "GraphGroupRelationRequestV2",
    "GraphEvidenceRefV2",
    "GraphEvidenceAssociationV2",
    "BoundedGraphTopologyV2",
]
