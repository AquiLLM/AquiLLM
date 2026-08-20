# ruff: noqa: E501, E701, E702, E704
# fmt: off
"""Closed opaque-key contract for provider-neutral graph ranking snapshots."""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import field, fields, make_dataclass
from enum import StrEnum
from hashlib import sha256
from math import isfinite
from typing import Literal

from apps.knowledge_graph.projection.serialization import _count, _key, _token

_INTS = frozenset("chunk_number admission_hop build_generation orchestration_version discovery_hop load_max_hops max_seeds max_scope_documents max_scope_collections max_hops max_nodes max_edges max_evidence_rows max_evidence_per_edge max_mentions_per_entity".split())
_NESTED = frozenset("orientation direction scope_type evaluation_only evidence signature document_keys collection_keys algorithm caps allowed_scope identity_keys seed_identities relation_groups mentions artifact_provenance audit_rows".split())
_RELATION_TYPE_PATTERN = re.compile(r"[a-z][a-z0-9_]{0,127}")
_ALGORITHM_TAGS = {"algorithm_version": "ppr_projected_v1", "transition_version": "ppr_transition_v1", "evidence_version": "ppr_evidence_v1", "seed_version": "rrf_seed_v1"}

def _rows(value: object, kind: type, name: str, cap: int, order) -> tuple:
    if type(value) is not tuple or any(type(row) is not kind for row in value): raise TypeError(f"{name} must contain exact {kind.__name__} values")
    keys = tuple(order(row) for row in value)
    if len(value) > cap or len(set(keys)) != len(keys): raise ValueError(f"{name} exceeds its cap or contains duplicates")
    if keys != tuple(sorted(keys)): raise ValueError(f"{name} must be canonically sorted")
    return value

ProjectedEvidenceOrientationV1 = StrEnum("ProjectedEvidenceOrientationV1", {"HEAD_TO_TAIL": "head_to_tail", "TAIL_TO_HEAD": "tail_to_head"})
ProjectedRetrievalDirectionV1 = StrEnum("ProjectedRetrievalDirectionV1", {"FORWARD": "forward", "REVERSE_DIRECTED": "reverse_directed", "UNDIRECTED": "undirected"})
ProjectedScopeTypeV1 = StrEnum("ProjectedScopeTypeV1", {"COLLECTION": "collection", "DOCUMENT": "document"})

class _Record:
    __slots__ = ()
    _finalized = False
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if getattr(cls, "_finalized", False): raise TypeError("projected DTO classes are final")
    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if value is None and item.name == "rebuild_request_key": continue
            if item.name == "embedding_model_signature":
                if type(value) is not str: raise TypeError("embedding_model_signature must be an exact str")
                if value: _token(value, item.name)
            elif item.name == "relation_type":
                if type(value) is not str: raise TypeError("relation_type must be an exact str")
                if _RELATION_TYPE_PATTERN.fullmatch(value) is None: raise ValueError("relation_type must be an exact canonical token")
            elif item.name.endswith(("_key", "_checksum", "_signature")) or item.name == "source_hash": _key(value, item.name)
            elif item.name in _INTS: _count(value, item.name)
            elif item.name in {"confidence", "raw_weight"}:
                if type(value) is not float or not isfinite(value): raise TypeError(f"{item.name} must be a finite built-in float")
                if item.name == "confidence" and not 0.0 <= value <= 1.0: raise ValueError("confidence must be in [0, 1]")
                if item.name == "raw_weight" and value <= 0.0: raise ValueError("raw_weight must be positive")
            elif type(value) is str: _token(value, item.name)
            elif item.name not in _NESTED: raise TypeError(f"{item.name} has an unsupported exact type")
        _SPECIAL.get(type(self), _noop)(self)

def _dto(name: str, names: str, overrides=None, tag: str | None = None):
    overrides = overrides or {}; definitions = [(item, overrides.get(item, str)) for item in names.split()]
    if tag is not None: definitions.append(("kind", Literal[tag], field(init=False, default=tag)))
    result = make_dataclass(name, definitions, bases=(_Record,), namespace={"__module__": __name__}, frozen=True, slots=True); result._finalized = True
    return result

ProjectedEvidenceSignatureV1 = _dto(
    "ProjectedEvidenceSignatureV1",
    "evidence_key relation_key relation_mention_key chunk_key document_key chunk_number confidence artifact_key source_document_key head_mention_key tail_mention_key relation_type head_mapping_key tail_mapping_key orientation ontology_checksum assembly_config_checksum",
    {"chunk_number": int, "confidence": float, "orientation": ProjectedEvidenceOrientationV1},
)
ProjectedChunkEvidenceV1 = _dto("ProjectedChunkEvidenceV1", "chunk_key document_key chunk_number confidence provenance_key", {"chunk_number": int, "confidence": float})
ProjectedSeedIdentityV1 = _dto("ProjectedSeedIdentityV1", "seed_chunk_key identity_key")
ProjectedRelationGroupV1 = _dto(
    "ProjectedRelationGroupV1", "source_identity_key relation_type target_identity_key direction raw_weight admission_hop evidence",
    {"direction": ProjectedRetrievalDirectionV1, "raw_weight": float, "admission_hop": int, "evidence": tuple[ProjectedChunkEvidenceV1, ...]},
)
ProjectedIdentityMentionV1 = _dto("ProjectedIdentityMentionV1", "identity_key evidence", {"evidence": ProjectedChunkEvidenceV1})
ProjectedArtifactProvenanceV1 = _dto(
    "ProjectedArtifactProvenanceV1",
    "artifact_key scope_type scope_key collection_key rebuild_request_key evaluation_only build_key build_generation orchestration_version source_hash ontology_version ontology_checksum extractor_version resolver_version resolution_config_checksum filter_policy_version filter_policy_checksum embedding_model_signature assembly_version assembly_config_checksum",
    {"scope_type": ProjectedScopeTypeV1, "rebuild_request_key": str | None, "evaluation_only": bool, "build_generation": int, "orchestration_version": int},
)
ProjectedAllowedScopeV1 = _dto("ProjectedAllowedScopeV1", "document_keys collection_keys scope_version_signature", {"document_keys": tuple[str, ...], "collection_keys": tuple[str, ...]})
ProjectedAlgorithmSignatureV1 = _dto("ProjectedAlgorithmSignatureV1", "algorithm_version transition_version evidence_version seed_version algorithm_signature")
ProjectedSnapshotCapsV1 = _dto(
    "ProjectedSnapshotCapsV1", "max_seeds max_scope_documents max_scope_collections max_hops max_nodes max_edges max_evidence_rows max_evidence_per_edge max_mentions_per_entity",
    {name: int for name in _INTS if name.startswith("max_")},
)
ProjectedAutomaticMembershipAuditV1 = _dto("ProjectedAutomaticMembershipAuditV1", "discovery_hop entity_key automatic_membership_key decision_checksum resolver_version", {"discovery_hop": int}, "automatic_membership")
ProjectedPhysicalRelationAuditV1 = _dto("ProjectedPhysicalRelationAuditV1", "discovery_hop relation_key artifact_key source_entity_key relation_type target_entity_key", {"discovery_hop": int}, "physical_relation")
ProjectedRelationEvidenceAuditV1 = _dto("ProjectedRelationEvidenceAuditV1", "discovery_hop signature", {"discovery_hop": int, "signature": ProjectedEvidenceSignatureV1}, "relation_evidence")
ProjectedFallbackMentionAuditV1 = _dto("ProjectedFallbackMentionAuditV1", "discovery_hop identity_key evidence", {"discovery_hop": int, "evidence": ProjectedChunkEvidenceV1}, "fallback_mention")
_AUDITS = (ProjectedAutomaticMembershipAuditV1, ProjectedFallbackMentionAuditV1, ProjectedPhysicalRelationAuditV1, ProjectedRelationEvidenceAuditV1)
ProjectedAuditRowV1 = ProjectedAutomaticMembershipAuditV1 | ProjectedFallbackMentionAuditV1 | ProjectedPhysicalRelationAuditV1 | ProjectedRelationEvidenceAuditV1
ProjectedAuthorizedGraphSnapshotV1 = _dto(
    "ProjectedAuthorizedGraphSnapshotV1", "algorithm caps load_max_hops allowed_scope identity_keys seed_identities relation_groups mentions artifact_provenance audit_rows",
    {"algorithm": ProjectedAlgorithmSignatureV1, "caps": ProjectedSnapshotCapsV1, "load_max_hops": int, "allowed_scope": ProjectedAllowedScopeV1, "identity_keys": tuple[str, ...], "seed_identities": tuple[ProjectedSeedIdentityV1, ...], "relation_groups": tuple[ProjectedRelationGroupV1, ...], "mentions": tuple[ProjectedIdentityMentionV1, ...], "artifact_provenance": tuple[ProjectedArtifactProvenanceV1, ...], "audit_rows": tuple[ProjectedAuditRowV1, ...]},
)

def _noop(value) -> None: del value
def _exact(value, kind, name):
    if type(value) is not kind: raise TypeError(f"{name} must be exact")
def _signature(value) -> None:
    if type(value.orientation) is not ProjectedEvidenceOrientationV1: raise TypeError("orientation must be exact")
    if value.source_document_key != value.document_key: raise ValueError("source_document_key must equal document_key")
def _group(value) -> None:
    if value.source_identity_key == value.target_identity_key: raise ValueError("relation group cannot be a self-loop")
    if type(value.direction) is not ProjectedRetrievalDirectionV1: raise TypeError("direction must be exact")
    _count(value.admission_hop, "admission_hop", 1, 2); _rows(value.evidence, ProjectedChunkEvidenceV1, "evidence", 3_000, _evidence_key)
def _provenance(value) -> None:
    if type(value.scope_type) is not ProjectedScopeTypeV1: raise TypeError("scope_type must be exact")
    if type(value.evaluation_only) is not bool: raise TypeError("evaluation_only must be exact")
    _count(value.build_generation, "build_generation", 1); _count(value.orchestration_version, "orchestration_version", 1)
    if (value.scope_type is ProjectedScopeTypeV1.COLLECTION) != bool(value.embedding_model_signature): raise ValueError("embedding signature disagrees with scope_type")
def _scope(value) -> None:
    _rows(value.document_keys, str, "document_keys", 10_000, _opaque_key); _rows(value.collection_keys, str, "collection_keys", 128, _opaque_key)
    if not value.document_keys or not value.collection_keys: raise ValueError("allowed scope must not be empty")
    if len(value.collection_keys) > len(value.document_keys): raise ValueError("allowed scope is incoherent")
def _caps(value) -> None:
    for item, maximum in zip(fields(value), (64, 10_000, 128, 2, 200, 1_000, 3_000, 3, 2), strict=True): _count(getattr(value, item.name), item.name, 1, maximum)
    if value.max_scope_collections > value.max_scope_documents or value.max_evidence_per_edge > value.max_evidence_rows: raise ValueError("snapshot caps are incoherent")
def _algorithm(value) -> None:
    for name, expected in _ALGORITHM_TAGS.items():
        if getattr(value, name) != expected: raise ValueError(f"{name} must be {expected}")
def _audit(value) -> None:
    low = 1 if type(value) in {ProjectedPhysicalRelationAuditV1, ProjectedRelationEvidenceAuditV1} else 0; _count(value.discovery_hop, "discovery_hop", low, 2)
    if type(value) is ProjectedPhysicalRelationAuditV1 and value.source_entity_key == value.target_entity_key: raise ValueError("physical relation cannot be a self-loop")
    if type(value) is ProjectedRelationEvidenceAuditV1: _exact(value.signature, ProjectedEvidenceSignatureV1, "signature")
    if type(value) is ProjectedFallbackMentionAuditV1: _exact(value.evidence, ProjectedChunkEvidenceV1, "evidence")

_SPECIAL = {
    ProjectedEvidenceSignatureV1: _signature, ProjectedRelationGroupV1: _group,
    ProjectedIdentityMentionV1: lambda value: _exact(value.evidence, ProjectedChunkEvidenceV1, "evidence"),
    ProjectedArtifactProvenanceV1: _provenance, ProjectedAllowedScopeV1: _scope,
    ProjectedAlgorithmSignatureV1: _algorithm, ProjectedSnapshotCapsV1: _caps, **{kind: _audit for kind in _AUDITS},
}

def _validate_snapshot(s) -> None:
    _exact(s.algorithm, ProjectedAlgorithmSignatureV1, "algorithm"); _exact(s.caps, ProjectedSnapshotCapsV1, "caps"); _exact(s.allowed_scope, ProjectedAllowedScopeV1, "allowed_scope")
    _count(s.load_max_hops, "load_max_hops", 1, s.caps.max_hops); _rows(s.identity_keys, str, "identities", s.caps.max_nodes, _opaque_key)
    if not s.identity_keys: raise ValueError("identity_keys must not be empty")
    specs = (
        (s.seed_identities, ProjectedSeedIdentityV1, s.caps.max_nodes, _seed_key),
        (s.relation_groups, ProjectedRelationGroupV1, s.caps.max_edges * 2, _group_key),
        (s.mentions, ProjectedIdentityMentionV1, s.caps.max_nodes * s.caps.max_mentions_per_entity, _mention_key),
        (s.artifact_provenance, ProjectedArtifactProvenanceV1, s.caps.max_scope_documents + s.caps.max_scope_collections, _provenance_key),
    )
    for rows, kind, cap, order in specs: _rows(rows, kind, kind.__name__, cap, order)
    if len({row.seed_chunk_key for row in s.seed_identities}) > s.caps.max_seeds: raise ValueError("distinct seed chunks exceed max_seeds")
    audit_cap = s.caps.max_nodes + s.caps.max_edges + s.caps.max_evidence_rows + s.caps.max_nodes * s.caps.max_mentions_per_entity
    if type(s.audit_rows) is not tuple or len(s.audit_rows) > audit_cap or any(type(row) not in _AUDITS for row in s.audit_rows): raise TypeError("audit_rows violate their closed union/cap")
    family_caps = {ProjectedAutomaticMembershipAuditV1: s.caps.max_nodes, ProjectedPhysicalRelationAuditV1: s.caps.max_edges, ProjectedRelationEvidenceAuditV1: s.caps.max_evidence_rows, ProjectedFallbackMentionAuditV1: s.caps.max_nodes * s.caps.max_mentions_per_entity}
    if any(sum(type(row) is kind for row in s.audit_rows) > cap for kind, cap in family_caps.items()): raise ValueError("audit family exceeds its hard cap")
    audit_keys = tuple((row.discovery_hop, row.kind, _canonical(row)) for row in s.audit_rows)
    if len(set(audit_keys)) != len(audit_keys) or audit_keys != tuple(sorted(audit_keys)): raise ValueError("audit_rows must be unique and canonically sorted")
    _validate_closure(s)

def _validate_closure(s) -> None:
    documents = set(s.allowed_scope.document_keys); collections = set(s.allowed_scope.collection_keys)
    if len(documents) > s.caps.max_scope_documents or len(collections) > s.caps.max_scope_collections: raise ValueError("scope exceeds caps")
    memberships = [row for row in s.audit_rows if type(row) is ProjectedAutomaticMembershipAuditV1]; members = {row.entity_key: row.automatic_membership_key for row in memberships}
    if len(members) != len(memberships) or set(members.values()) != set(s.identity_keys): raise ValueError("automatic membership/identity closure is broken")
    if any(row.discovery_hop > s.load_max_hops for row in s.audit_rows): raise ValueError("audit rows exceed load_max_hops")
    if any(row.identity_key not in s.identity_keys for row in s.seed_identities): raise ValueError("seed identity closure is broken")
    artifacts = {row.artifact_key: row for row in s.artifact_provenance}
    collection_rows = [row for row in s.artifact_provenance if row.scope_type is ProjectedScopeTypeV1.COLLECTION]; document_rows = [row for row in s.artifact_provenance if row.scope_type is ProjectedScopeTypeV1.DOCUMENT]
    collection_artifacts = {row.artifact_key: row for row in collection_rows}; document_artifacts = {row.artifact_key: row for row in document_rows}
    if len(artifacts) != len(s.artifact_provenance) or len(collection_rows) != len(collections) or len(document_rows) != len(documents) or {row.scope_key for row in collection_rows} != collections or {row.scope_key for row in document_rows} != documents: raise ValueError("artifact provenance scope closure is broken")
    if any(row.collection_key not in collections or (row in collection_rows and row.scope_key != row.collection_key) for row in s.artifact_provenance): raise ValueError("artifact provenance collection closure is broken")
    physical_rows = [row for row in s.audit_rows if type(row) is ProjectedPhysicalRelationAuditV1]; physical = {row.relation_key: row for row in physical_rows}
    if len(physical) != len(physical_rows) or any(row.source_entity_key not in members or row.target_entity_key not in members or row.artifact_key not in collection_artifacts for row in physical.values()): raise ValueError("physical relation/member closure is broken")
    evidence_audits = [row for row in s.audit_rows if type(row) is ProjectedRelationEvidenceAuditV1]; signature_rows = [row.signature for row in evidence_audits]; signatures = {row.evidence_key: row for row in signature_rows}
    if len(signatures) != len(signature_rows) or len(signatures) > s.caps.max_evidence_rows: raise ValueError("evidence keys/cap are invalid")
    if any(sig.relation_key not in physical or sig.document_key not in documents or sig.artifact_key not in document_artifacts or document_artifacts[sig.artifact_key].scope_key != sig.document_key for sig in signatures.values()): raise ValueError("relation evidence closure is broken")
    if any(sig.relation_type != physical[sig.relation_key].relation_type or document_artifacts[sig.artifact_key].collection_key != collection_artifacts[physical[sig.relation_key].artifact_key].scope_key or sig.ontology_checksum != collection_artifacts[physical[sig.relation_key].artifact_key].ontology_checksum or sig.assembly_config_checksum != collection_artifacts[physical[sig.relation_key].artifact_key].assembly_config_checksum for sig in signatures.values()): raise ValueError("relation evidence semantic/provenance closure is broken")
    if any(row.discovery_hop != physical[row.signature.relation_key].discovery_hop for row in evidence_audits): raise ValueError("relation evidence discovery hop is incoherent")
    semantic: dict[tuple[str, str, str], set[str]] = {}
    for row in physical.values(): semantic.setdefault((members[row.source_entity_key], row.relation_type, members[row.target_entity_key]), set()).add(row.relation_key)
    if set(physical) != {row.relation_key for row in signatures.values()}: raise ValueError("physical relation/evidence coverage is broken")
    observed: set[str] = set(); references: dict[str, list[ProjectedRetrievalDirectionV1]] = {}; coordinates: dict[str, tuple[str, int]] = {}
    for group in s.relation_groups:
        relation_keys = _group_relations(group, semantic, s)
        for evidence in group.evidence:
            signature = signatures.get(evidence.provenance_key); actual = (evidence.chunk_key, evidence.document_key, evidence.chunk_number, evidence.confidence)
            if signature is None or signature.relation_key not in relation_keys or _signature_coordinate(signature) != actual: raise ValueError("group evidence/signature closure is broken")
            coordinate = actual[1:3]
            if coordinates.setdefault(evidence.chunk_key, coordinate) != coordinate: raise ValueError("chunk coordinate closure is broken")
            observed.add(evidence.provenance_key); references.setdefault(evidence.provenance_key, []).append(group.direction)
    if observed != set(signatures): raise ValueError("relation evidence coverage is broken")
    for directions in references.values():
        if len(directions) > 2 or (ProjectedRetrievalDirectionV1.UNDIRECTED in directions and any(item is not ProjectedRetrievalDirectionV1.UNDIRECTED for item in directions)): raise ValueError("evidence reference direction set is incoherent")
    fallback_rows = [row for row in s.audit_rows if type(row) is ProjectedFallbackMentionAuditV1]; fallback_keys = [(row.identity_key, row.evidence.provenance_key) for row in fallback_rows]
    if len(set(fallback_keys)) != len(fallback_keys): raise ValueError("fallback semantic key is duplicated")
    identity_hops: dict[str, int] = {}
    for row in memberships: identity_hops[row.automatic_membership_key] = min(row.discovery_hop, identity_hops.get(row.automatic_membership_key, row.discovery_hop))
    if any(identity_hops.get(row.identity_key) != row.discovery_hop for row in fallback_rows): raise ValueError("fallback discovery hop is incoherent")
    fallback = {key: row.evidence for key, row in zip(fallback_keys, fallback_rows, strict=True)}; mentions = {(row.identity_key, row.evidence.provenance_key): row.evidence for row in s.mentions}
    if mentions != fallback: raise ValueError("fallback mention closure is broken")
    counts = Counter(row.identity_key for row in s.mentions)
    for mention in s.mentions:
        if mention.identity_key not in s.identity_keys or mention.evidence.document_key not in documents or counts[mention.identity_key] > s.caps.max_mentions_per_entity: raise ValueError("mention cap/scope closure is broken")
        coordinate = mention.evidence.document_key, mention.evidence.chunk_number
        if coordinates.setdefault(mention.evidence.chunk_key, coordinate) != coordinate: raise ValueError("chunk coordinate closure is broken")

def _group_relations(group, semantic, snapshot):
    if group.admission_hop > snapshot.load_max_hops or len(group.evidence) > snapshot.caps.max_evidence_per_edge: raise ValueError("relation group exceeds caps")
    key = group.source_identity_key, group.relation_type, group.target_identity_key
    if group.direction is ProjectedRetrievalDirectionV1.REVERSE_DIRECTED: key = key[2], key[1], key[0]
    relations = semantic.get(key)
    if relations is None and group.direction is ProjectedRetrievalDirectionV1.UNDIRECTED: relations = semantic.get((key[2], key[1], key[0]))
    if relations is None: raise ValueError("relation group/member closure is broken")
    return relations

def _opaque_key(row): _key(row, "opaque_key"); return row
def _evidence_key(row): return row.provenance_key, row.chunk_key
def _seed_key(row): return row.seed_chunk_key, row.identity_key
def _group_key(row): return row.source_identity_key, row.relation_type, row.target_identity_key, row.direction.value
def _mention_key(row): return row.identity_key, row.evidence.provenance_key
def _provenance_key(row): return row.scope_type.value, row.scope_key, row.artifact_key
def _signature_coordinate(row): return row.chunk_key, row.document_key, row.chunk_number, row.confidence

_CANONICAL = frozenset({ProjectedEvidenceSignatureV1, ProjectedChunkEvidenceV1, ProjectedSeedIdentityV1, ProjectedRelationGroupV1, ProjectedIdentityMentionV1, ProjectedArtifactProvenanceV1, ProjectedAllowedScopeV1, ProjectedAlgorithmSignatureV1, ProjectedSnapshotCapsV1, *_AUDITS, ProjectedAuthorizedGraphSnapshotV1})
_ENUMS = (ProjectedEvidenceOrientationV1, ProjectedRetrievalDirectionV1, ProjectedScopeTypeV1)
def _encode(value: object) -> object:
    if type(value) in _CANONICAL: return {item.name: _encode(getattr(value, item.name)) for item in fields(value)}
    if type(value) is tuple: return [_encode(item) for item in value]
    if type(value) in _ENUMS: return value.value
    if type(value) is float: return value.hex()
    if value is None or type(value) in {str, int, bool}: return value
    raise TypeError("snapshot contains an unsupported canonical value")
def _canonical(value: object) -> str: return json.dumps(_encode(value), ensure_ascii=False, separators=(",", ":"), sort_keys=True)
def canonical_projected_snapshot_bytes(snapshot) -> bytes:
    _exact(snapshot, ProjectedAuthorizedGraphSnapshotV1, "snapshot"); return _canonical(snapshot).encode("utf-8")
def projected_snapshot_checksum(snapshot) -> str: return sha256(canonical_projected_snapshot_bytes(snapshot)).hexdigest()
_SPECIAL[ProjectedAuthorizedGraphSnapshotV1] = _validate_snapshot
