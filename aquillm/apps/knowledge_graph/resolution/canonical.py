"""Conservative cross-collection identity resolution and permission projection.

Canonical identities are an internal registry.  The permission-bearing rows are
always active collection entities backed by active automatic document mappings.
"""

from __future__ import annotations

import json
import re
import uuid
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from itertools import combinations
from math import isfinite

from apps.knowledge_graph.resolution.normalization import (
    normalize_entity_label,
    parse_stable_identifier,
)

CANONICAL_RESOLVER_VERSION = "canonical-resolution-v1"

MAX_CANONICAL_COLLECTIONS = 128
MAX_CANONICAL_ARTIFACTS = 128
MAX_CANONICAL_ENTITIES = 10_000
MAX_CANONICAL_SOURCE_LINKS = 30_000
MAX_CANONICAL_MEMBERSHIPS = 50_000
MAX_CANONICAL_DECISIONS = 50_000
CANONICAL_QUERY_BATCH_SIZE = 5_000
CANONICAL_EMBEDDING_CANDIDATE_MIN_SIMILARITY = 0.88
CANONICAL_EMBEDDING_PROJECTION_WINDOW = 4

_HASH_PATTERN = re.compile(r"[0-9a-f]{64}")
_VERSION_PATTERN = re.compile(r"[a-z0-9][a-z0-9.+:/_-]*")
_ACRONYM_PATTERN = re.compile(r"[A-Z][A-Z0-9-]{1,11}")
_METHOD_PRIORITY = {
    "stable_identifier": 0,
    "exact_name_or_alias": 1,
    "defined_acronym": 2,
    "embedding_similarity": 3,
}


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or not 1 <= value <= 2**63 - 1:
        raise ValueError(f"{label} must be a positive database integer")
    return value


def _bounded_text(
    value: object,
    label: str,
    *,
    maximum: int,
    allow_empty: bool = False,
) -> str:
    if type(value) is not str or value != value.strip() or "\x00" in value:
        raise ValueError(f"{label} must be an exact trimmed string")
    if (not value and not allow_empty) or len(value) > maximum:
        emptiness = "possibly empty" if allow_empty else "nonempty"
        raise ValueError(f"{label} must be a bounded {emptiness} string")
    return value


def _hash_payload(payload: object) -> str:
    return sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _normalized_key(value: str) -> str:
    return normalize_entity_label(value).key


def _is_acronym(value: str) -> bool:
    return bool(_ACRONYM_PATTERN.fullmatch(value))


class CanonicalOutcome(StrEnum):
    AUTOMATIC = "automatic"
    CANDIDATE = "candidate"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class CanonicalAliasEvidence:
    """One exact Task-8 ontology-alias assignment with parent provenance."""

    alias: str
    method: str
    document_entity_id: int
    mention_id: int
    parent_mention_id: int
    normalized_alias: str = ""
    is_acronym: bool = False

    def __post_init__(self) -> None:
        raw = _bounded_text(self.alias, "alias", maximum=512)
        if self.method != "ontology_alias":
            raise ValueError("canonical aliases require exact ontology_alias evidence")
        _positive_int(self.document_entity_id, "alias document_entity_id")
        _positive_int(self.mention_id, "alias mention_id")
        _positive_int(self.parent_mention_id, "alias parent_mention_id")
        if self.mention_id == self.parent_mention_id:
            raise ValueError("alias evidence must name a distinct parent mention")
        object.__setattr__(self, "normalized_alias", _normalized_key(raw))
        object.__setattr__(self, "is_acronym", _is_acronym(raw))


@dataclass(frozen=True, slots=True)
class CanonicalAcronymExpansion:
    """One Task-8 defined-acronym assignment bound to its two mentions."""

    acronym: str
    full_form: str
    document_entity_id: int
    acronym_mention_id: int
    full_form_mention_id: int

    def __post_init__(self) -> None:
        acronym = _bounded_text(self.acronym, "acronym", maximum=64)
        full_form = _bounded_text(self.full_form, "acronym full form", maximum=512)
        if not _is_acronym(acronym):
            raise ValueError("acronym surface must use the uppercase acronym grammar")
        if _is_acronym(full_form) or _normalized_key(acronym) == _normalized_key(
            full_form
        ):
            raise ValueError("acronym full form must be a distinct non-acronym surface")
        _positive_int(self.document_entity_id, "acronym document_entity_id")
        _positive_int(self.acronym_mention_id, "acronym mention_id")
        _positive_int(self.full_form_mention_id, "full-form mention_id")
        if self.acronym_mention_id == self.full_form_mention_id:
            raise ValueError("acronym evidence must bind two distinct mentions")
        object.__setattr__(self, "acronym", _normalized_key(acronym))
        object.__setattr__(self, "full_form", _normalized_key(full_form))


@dataclass(frozen=True, slots=True)
class CanonicalProvenanceRow:
    """Exact active Task-8 evidence projected through a Task-9 automatic link."""

    collection_entity_id: int
    document_entity_id: int
    document_artifact_id: int
    mention_id: int
    parent_mention_id: int
    method: str
    surface: str
    parent_surface: str
    source_collection_entity_id: int

    def __post_init__(self) -> None:
        for field_name in (
            "collection_entity_id",
            "document_entity_id",
            "document_artifact_id",
            "mention_id",
            "parent_mention_id",
            "source_collection_entity_id",
        ):
            _positive_int(getattr(self, field_name), field_name)
        if self.mention_id == self.parent_mention_id:
            raise ValueError("provenance child and parent mentions must differ")
        if self.method not in {"ontology_alias", "defined_acronym"}:
            raise ValueError("canonical provenance method is not eligible")
        _bounded_text(self.surface, "provenance surface", maximum=4_096)
        _bounded_text(self.parent_surface, "provenance parent surface", maximum=4_096)


@dataclass(frozen=True, slots=True)
class CanonicalSourceMembership:
    """Exact current Task-9 automatic collection-to-document membership."""

    collection_entity_id: int
    document_entity_id: int
    document_artifact_id: int

    def __post_init__(self) -> None:
        _positive_int(self.collection_entity_id, "source collection_entity_id")
        _positive_int(self.document_entity_id, "source document_entity_id")
        _positive_int(self.document_artifact_id, "source document_artifact_id")


@dataclass(frozen=True, slots=True)
class CanonicalEntityInput:
    """One current collection entity with exact, already-audited evidence."""

    entity_id: int
    collection_id: int
    artifact_id: int
    cluster_key: str
    label: str
    normalized_label: str
    entity_type: str
    identifier: str = ""
    version_signature: str = ""
    aliases: tuple[CanonicalAliasEvidence, ...] = ()
    acronym_expansions: tuple[CanonicalAcronymExpansion, ...] = ()

    def __post_init__(self) -> None:
        _positive_int(self.entity_id, "entity_id")
        _positive_int(self.collection_id, "collection_id")
        _positive_int(self.artifact_id, "artifact_id")
        if type(self.cluster_key) is not str or not _HASH_PATTERN.fullmatch(
            self.cluster_key
        ):
            raise ValueError("cluster_key must be lowercase SHA-256")
        _bounded_text(self.label, "label", maximum=4_096)
        normalized = _bounded_text(
            self.normalized_label, "normalized_label", maximum=512
        )
        expected_normalized = _normalized_key(self.label)
        if normalized != expected_normalized:
            raise ValueError("normalized_label must exactly match the normalized label")
        object.__setattr__(self, "normalized_label", expected_normalized)
        _bounded_text(self.entity_type, "entity_type", maximum=128)
        identifier = _bounded_text(
            self.identifier, "identifier", maximum=255, allow_empty=True
        )
        if identifier:
            parsed = parse_stable_identifier(identifier)
            if parsed is None or parsed.canonical != identifier:
                raise ValueError("stable identifier must be an exact canonical value")
        version = _bounded_text(
            self.version_signature,
            "version_signature",
            maximum=128,
            allow_empty=True,
        )
        if version and not _VERSION_PATTERN.fullmatch(version):
            raise ValueError("version_signature must use canonical tokens")
        if type(self.aliases) is not tuple:
            raise ValueError("aliases must be an exact tuple")
        if any(type(alias) is not CanonicalAliasEvidence for alias in self.aliases):
            raise ValueError("aliases must contain exact CanonicalAliasEvidence values")
        aliases = tuple(
            sorted(
                self.aliases,
                key=lambda item: (
                    item.normalized_alias,
                    item.document_entity_id,
                    item.mention_id,
                ),
            )
        )
        alias_keys = {(item.normalized_alias, item.mention_id) for item in aliases}
        if len(alias_keys) != len(aliases):
            raise ValueError("alias evidence must be unique")
        object.__setattr__(self, "aliases", aliases)
        if type(self.acronym_expansions) is not tuple:
            raise ValueError("acronym_expansions must be an exact tuple")
        if any(
            type(binding) is not CanonicalAcronymExpansion
            for binding in self.acronym_expansions
        ):
            raise ValueError(
                "acronym expansions must contain exact CanonicalAcronymExpansion values"
            )
        normalized_expansions = tuple(
            sorted(
                self.acronym_expansions,
                key=lambda item: (
                    item.acronym,
                    item.full_form,
                    item.document_entity_id,
                    item.acronym_mention_id,
                ),
            )
        )
        expansion_keys = {
            (
                item.acronym,
                item.full_form,
                item.document_entity_id,
                item.acronym_mention_id,
                item.full_form_mention_id,
            )
            for item in normalized_expansions
        }
        if len(normalized_expansions) != len(expansion_keys):
            raise ValueError("acronym expansions must be unique")
        object.__setattr__(self, "acronym_expansions", normalized_expansions)


def build_canonical_inputs_from_provenance(
    entity_rows: tuple[object, ...],
    provenance_rows: tuple[CanonicalProvenanceRow, ...],
    source_memberships: tuple[CanonicalSourceMembership, ...],
) -> tuple[CanonicalEntityInput, ...]:
    """Adapt locked ORM rows without trusting unbound CollectionEntity metadata."""

    if type(entity_rows) is not tuple or len(entity_rows) > MAX_CANONICAL_ENTITIES:
        raise ValueError("entity rows exceed the canonical input envelope")
    if type(provenance_rows) is not tuple or any(
        type(row) is not CanonicalProvenanceRow for row in provenance_rows
    ):
        raise ValueError("provenance rows must be an exact typed tuple")
    if len(provenance_rows) > MAX_CANONICAL_MEMBERSHIPS:
        raise ValueError("provenance rows exceed the canonical membership cap")
    if type(source_memberships) is not tuple or any(
        type(row) is not CanonicalSourceMembership for row in source_memberships
    ):
        raise ValueError("source memberships must be an exact typed tuple")
    if len(source_memberships) > MAX_CANONICAL_SOURCE_LINKS:
        raise ValueError("source memberships exceed the canonical source-link cap")
    by_id: dict[int, object] = {}
    for entity in entity_rows:
        entity_id = _positive_int(getattr(entity, "pk", None), "entity row id")
        if entity_id in by_id:
            raise ValueError("entity rows contain a duplicate primary key")
        by_id[entity_id] = entity
    source_keys = {
        (
            row.collection_entity_id,
            row.document_entity_id,
            row.document_artifact_id,
        )
        for row in source_memberships
    }
    if len(source_keys) != len(source_memberships):
        raise ValueError("source memberships contain a duplicate exact mapping")
    if any(row.collection_entity_id not in by_id for row in source_memberships):
        raise ValueError("source membership escaped the exact entity snapshot")
    provenance_by_entity: dict[int, list[CanonicalProvenanceRow]] = defaultdict(list)
    for row in provenance_rows:
        if row.collection_entity_id not in by_id:
            raise ValueError("provenance escaped the exact collection entity snapshot")
        if row.source_collection_entity_id != row.collection_entity_id:
            raise ValueError(
                "provenance does not belong to the exact collection entity"
            )
        if (
            row.collection_entity_id,
            row.document_entity_id,
            row.document_artifact_id,
        ) not in source_keys:
            raise ValueError("provenance escaped the exact active source membership")
        provenance_by_entity[row.collection_entity_id].append(row)

    inputs: list[CanonicalEntityInput] = []
    for entity_id, entity in sorted(by_id.items()):
        aliases: list[CanonicalAliasEvidence] = []
        expansions: list[CanonicalAcronymExpansion] = []
        for row in sorted(
            provenance_by_entity.get(entity_id, ()),
            key=lambda item: (
                item.document_entity_id,
                item.mention_id,
                item.parent_mention_id,
            ),
        ):
            if row.method == "ontology_alias":
                aliases.append(
                    CanonicalAliasEvidence(
                        alias=row.surface,
                        method=row.method,
                        document_entity_id=row.document_entity_id,
                        mention_id=row.mention_id,
                        parent_mention_id=row.parent_mention_id,
                    )
                )
                continue
            child_is_acronym = _is_acronym(row.surface)
            parent_is_acronym = _is_acronym(row.parent_surface)
            if child_is_acronym == parent_is_acronym:
                raise ValueError(
                    "defined acronym provenance must bind one acronym to one full form"
                )
            acronym_surface, acronym_mention_id = (
                (row.surface, row.mention_id)
                if child_is_acronym
                else (row.parent_surface, row.parent_mention_id)
            )
            full_surface, full_mention_id = (
                (row.parent_surface, row.parent_mention_id)
                if child_is_acronym
                else (row.surface, row.mention_id)
            )
            expansions.append(
                CanonicalAcronymExpansion(
                    acronym=acronym_surface,
                    full_form=full_surface,
                    document_entity_id=row.document_entity_id,
                    acronym_mention_id=acronym_mention_id,
                    full_form_mention_id=full_mention_id,
                )
            )
        inputs.append(
            CanonicalEntityInput(
                entity_id=entity_id,
                collection_id=getattr(entity, "collection_id"),
                artifact_id=getattr(entity, "artifact_id"),
                cluster_key=getattr(entity, "cluster_key"),
                label=getattr(entity, "label"),
                normalized_label=getattr(entity, "normalized_label"),
                entity_type=getattr(entity, "entity_type"),
                identifier=getattr(entity, "identifier"),
                version_signature=getattr(entity, "version_signature"),
                aliases=tuple(aliases),
                acronym_expansions=tuple(expansions),
            )
        )
    return tuple(inputs)


@dataclass(frozen=True, slots=True)
class CanonicalEmbeddingCandidate:
    """Audited same-model embedding proposal; never an automatic edge."""

    left_entity_id: int
    right_entity_id: int
    similarity: float
    embedding_model_signature: str
    left_input_hash: str
    right_input_hash: str

    def __post_init__(self) -> None:
        left = _positive_int(self.left_entity_id, "left_entity_id")
        right = _positive_int(self.right_entity_id, "right_entity_id")
        if left == right:
            raise ValueError("embedding candidate endpoints must differ")
        if type(self.similarity) not in (int, float):
            raise ValueError("embedding similarity must be finite in [0, 1]")
        similarity = float(self.similarity)
        if not isfinite(similarity) or not 0 <= similarity <= 1:
            raise ValueError("embedding similarity must be finite in [0, 1]")
        object.__setattr__(self, "similarity", similarity)
        _bounded_text(
            self.embedding_model_signature,
            "embedding_model_signature",
            maximum=512,
        )
        left_hash_valid = _HASH_PATTERN.fullmatch(self.left_input_hash or "")
        right_hash_valid = _HASH_PATTERN.fullmatch(self.right_input_hash or "")
        if not left_hash_valid or not right_hash_valid:
            raise ValueError("embedding input hashes must be lowercase SHA-256")
        if right < left:
            left_hash = self.left_input_hash
            right_hash = self.right_input_hash
            object.__setattr__(self, "left_entity_id", right)
            object.__setattr__(self, "right_entity_id", left)
            object.__setattr__(self, "left_input_hash", right_hash)
            object.__setattr__(self, "right_input_hash", left_hash)


@dataclass(frozen=True, slots=True)
class CanonicalDecision:
    left_entity_id: int
    right_entity_id: int
    score: float
    method: str
    outcome: CanonicalOutcome
    reason: str
    evidence_key: str = ""
    metadata: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class CanonicalComponent:
    identity_key: str
    entity_ids: tuple[int, ...]
    collection_ids: tuple[int, ...]
    label: str
    normalized_label: str
    entity_type: str
    version_signature: str
    method: str


@dataclass(frozen=True, slots=True)
class CanonicalResolutionResult:
    resolver_version: str
    components: tuple[CanonicalComponent, ...]
    decisions: tuple[CanonicalDecision, ...]
    checksum: str


@dataclass(frozen=True, slots=True)
class CanonicalRebuildResult:
    """Opaque audit summary for one committed full-registry reconciliation."""

    resolver_version: str
    resolution_checksum: str
    active_artifact_ids: tuple[int, ...]
    canonical_entity_ids: tuple[int, ...]
    active_link_count: int
    created_entity_count: int
    created_link_count: int
    superseded_entity_count: int
    superseded_link_count: int

    def __post_init__(self) -> None:
        _bounded_text(self.resolver_version, "resolver_version", maximum=128)
        if not _HASH_PATTERN.fullmatch(self.resolution_checksum or ""):
            raise ValueError("resolution_checksum must be lowercase SHA-256")
        _sorted_positive_tuple(
            self.active_artifact_ids,
            "active_artifact_ids",
            maximum=MAX_CANONICAL_ARTIFACTS,
        )
        _sorted_positive_tuple(
            self.canonical_entity_ids,
            "canonical_entity_ids",
            maximum=MAX_CANONICAL_ENTITIES,
        )
        for field_name in (
            "active_link_count",
            "created_entity_count",
            "created_link_count",
            "superseded_entity_count",
            "superseded_link_count",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or not 0 <= value <= MAX_CANONICAL_MEMBERSHIPS:
                raise ValueError(f"{field_name} exceeds the canonical audit envelope")


@dataclass(frozen=True, slots=True)
class CanonicalProjectedLinkDecision:
    """One deterministic source-to-registry audit ready for persistence."""

    source_entity_id: int
    target_identity_key: str
    score: float
    method: str
    outcome: CanonicalOutcome
    status: str
    reason: str
    metadata_json: str

    def __post_init__(self) -> None:
        _positive_int(self.source_entity_id, "projected source entity id")
        if not _HASH_PATTERN.fullmatch(self.target_identity_key or ""):
            raise ValueError("projected target identity must be lowercase SHA-256")
        if (
            type(self.score) not in (int, float)
            or not isfinite(self.score)
            or not 0 <= self.score <= 1
        ):
            raise ValueError("projected score must be finite in [0, 1]")
        expected_status = {
            CanonicalOutcome.AUTOMATIC: "active",
            CanonicalOutcome.CANDIDATE: "suppressed",
            CanonicalOutcome.REJECTED: "rejected",
        }.get(self.outcome)
        if self.status != expected_status:
            raise ValueError("projected status must correspond to outcome")
        _bounded_text(self.method, "projected method", maximum=64)
        _bounded_text(self.reason, "projected reason", maximum=4_096)
        try:
            metadata = json.loads(self.metadata_json)
        except (TypeError, ValueError) as exc:
            raise ValueError("projected metadata must be canonical JSON") from exc
        if type(metadata) is not dict or self.metadata_json != json.dumps(
            metadata,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ):
            raise ValueError("projected metadata must be a canonical JSON object")


class _DisjointSet:
    def __init__(self, values: Iterable[int]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: int) -> int:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while value != root:
            following = self.parent[value]
            self.parent[value] = root
            value = following
        return root

    def union(self, left: int, right: int) -> int:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return left_root
        root, child = sorted((left_root, right_root))
        self.parent[child] = root
        return root

    def groups(self) -> tuple[tuple[int, ...], ...]:
        grouped: dict[int, list[int]] = defaultdict(list)
        for value in sorted(self.parent):
            grouped[self.find(value)].append(value)
        return tuple(tuple(values) for _, values in sorted(grouped.items()))


@dataclass(frozen=True, slots=True)
class _EvidenceEdge:
    left: int
    right: int
    method: str
    evidence_key: str


def _pair(left: int, right: int) -> tuple[int, int]:
    return (left, right) if left < right else (right, left)


def _reserve_block_pairs(
    members: Sequence[int], reserved: set[tuple[int, int]]
) -> None:
    count = len(members)
    possible = count * (count - 1) // 2
    if possible > MAX_CANONICAL_DECISIONS + len(reserved):
        raise ValueError("canonical decision cap exceeded")
    for pair in combinations(sorted(members), 2):
        reserved.add(pair)
        if len(reserved) > MAX_CANONICAL_DECISIONS:
            raise ValueError("canonical decision cap exceeded")


def _entity_name_keys(entity: CanonicalEntityInput) -> frozenset[str]:
    # An acronym surface is never a global name.  It requires exact local
    # Task-8 definition provenance represented by acronym_expansions.
    keys: set[str] = set()
    if not _is_acronym(entity.label):
        keys.add(entity.normalized_label)
        keys.update(
            alias.normalized_alias for alias in entity.aliases if not alias.is_acronym
        )
    return frozenset(keys)


def _acronym_bindings(entity: CanonicalEntityInput) -> dict[str, frozenset[str]]:
    values: dict[str, set[str]] = defaultdict(set)
    for binding in entity.acronym_expansions:
        values[binding.acronym].add(binding.full_form)
    return {key: frozenset(items) for key, items in values.items()}


def _pair_conflict(
    left: CanonicalEntityInput, right: CanonicalEntityInput
) -> str | None:
    if left.collection_id == right.collection_id:
        return "same_collection_cannot_link"
    if left.entity_type != right.entity_type:
        return "ontology_type_conflict"
    if left.version_signature != right.version_signature:
        return "version_signature_conflict"
    return None


def _component_conflict(
    entity_ids: Sequence[int], by_id: Mapping[int, CanonicalEntityInput]
) -> str | None:
    members = tuple(by_id[value] for value in entity_ids)
    collection_ids = [item.collection_id for item in members]
    if len(collection_ids) != len(set(collection_ids)):
        return "component_same_collection_conflict"
    types = {item.entity_type for item in members}
    if len(types) > 1:
        return "component_ontology_type_conflict"
    versions = {item.version_signature for item in members}
    if len(versions) > 1:
        return "component_version_signature_conflict"
    identifiers = {item.identifier for item in members if item.identifier}
    if len(identifiers) > 1:
        return "component_stable_identifier_conflict"
    acronym_forms: dict[str, set[str]] = defaultdict(set)
    for item in members:
        for binding in item.acronym_expansions:
            acronym_forms[binding.acronym].add(binding.full_form)
    if any(len(forms) > 1 for forms in acronym_forms.values()):
        return "component_acronym_expansion_conflict"
    return None


def _rejected(
    left: int,
    right: int,
    *,
    method: str,
    reason: str,
    evidence_key: str,
) -> CanonicalDecision:
    return CanonicalDecision(
        *_pair(left, right),
        score=0.0,
        method=method,
        outcome=CanonicalOutcome.REJECTED,
        reason=reason,
        evidence_key=evidence_key,
    )


def _automatic(edge: _EvidenceEdge) -> CanonicalDecision:
    return CanonicalDecision(
        edge.left,
        edge.right,
        score={
            "stable_identifier": 1.0,
            "exact_name_or_alias": 0.99,
            "defined_acronym": 0.98,
        }[edge.method],
        method=edge.method,
        outcome=CanonicalOutcome.AUTOMATIC,
        reason={
            "stable_identifier": "identical_stable_identifier",
            "exact_name_or_alias": "exact_evidence_backed_name_or_alias",
            "defined_acronym": "identical_singleton_defined_acronym",
        }[edge.method],
        evidence_key=edge.evidence_key,
    )


def _connected_edge_groups(
    roots: Iterable[int], edges: Sequence[_EvidenceEdge]
) -> tuple[tuple[int, ...], ...]:
    dsu = _DisjointSet(roots)
    for edge in edges:
        dsu.union(edge.left, edge.right)
    return dsu.groups()


def _component_identity_payload(
    members: tuple[CanonicalEntityInput, ...],
    automatic_edges: tuple[_EvidenceEdge, ...],
) -> dict[str, object]:
    identifiers = sorted({item.identifier for item in members if item.identifier})
    if identifiers:
        return {
            "kind": "stable_identifier",
            "entity_type": members[0].entity_type,
            "identifier": identifiers[0],
            "version_signature": members[0].version_signature,
        }
    if automatic_edges:
        edge = min(
            automatic_edges,
            key=lambda item: (
                _METHOD_PRIORITY[item.method],
                item.evidence_key,
                item.left,
                item.right,
            ),
        )
        return {
            "kind": edge.method,
            "entity_type": members[0].entity_type,
            "evidence_key": edge.evidence_key,
            "version_signature": members[0].version_signature,
        }
    member = members[0]
    return {
        "kind": "singleton",
        "collection_id": member.collection_id,
        "cluster_key": member.cluster_key,
        "entity_type": member.entity_type,
        "identifier": member.identifier,
        "normalized_label": member.normalized_label,
        "version_signature": member.version_signature,
    }


def resolve_canonical_entities(
    entities: tuple[CanonicalEntityInput, ...],
    *,
    embedding_candidates: tuple[CanonicalEmbeddingCandidate, ...] = (),
    resolver_version: str = CANONICAL_RESOLVER_VERSION,
) -> CanonicalResolutionResult:
    """Resolve exact evidence into deterministic components and audit decisions."""

    _bounded_text(resolver_version, "resolver_version", maximum=128)
    if type(entities) is not tuple or any(
        type(item) is not CanonicalEntityInput for item in entities
    ):
        raise ValueError("entities must be an exact CanonicalEntityInput tuple")
    if len(entities) > MAX_CANONICAL_ENTITIES:
        raise ValueError("canonical entity cap exceeded")
    ordered = tuple(sorted(entities, key=lambda item: item.entity_id))
    by_id = {item.entity_id: item for item in ordered}
    if len(by_id) != len(ordered):
        raise ValueError("canonical entity IDs must be unique")
    if type(embedding_candidates) is not tuple or any(
        type(item) is not CanonicalEmbeddingCandidate for item in embedding_candidates
    ):
        raise ValueError("embedding candidates must be an exact typed tuple")
    if len(embedding_candidates) > MAX_CANONICAL_DECISIONS:
        raise ValueError("canonical candidate cap exceeded")

    rejected_decisions: dict[tuple[int, int], CanonicalDecision] = {}
    stable_edges: dict[tuple[int, int], _EvidenceEdge] = {}
    lower_edges: dict[tuple[int, int], _EvidenceEdge] = {}
    reserved_pairs: set[tuple[int, int]] = set()

    identifier_blocks: dict[str, list[int]] = defaultdict(list)
    for item in ordered:
        if item.identifier:
            identifier_blocks[item.identifier].append(item.entity_id)
    for identifier, members in sorted(identifier_blocks.items()):
        _reserve_block_pairs(members, reserved_pairs)
        for left_id, right_id in combinations(sorted(members), 2):
            left, right = by_id[left_id], by_id[right_id]
            conflict = _pair_conflict(left, right)
            if conflict:
                rejected_decisions[_pair(left_id, right_id)] = _rejected(
                    left_id,
                    right_id,
                    method="stable_identifier",
                    reason=conflict,
                    evidence_key=identifier,
                )
            else:
                stable_edges[_pair(left_id, right_id)] = _EvidenceEdge(
                    *_pair(left_id, right_id),
                    method="stable_identifier",
                    evidence_key=identifier,
                )

    name_blocks: dict[str, set[int]] = defaultdict(set)
    for item in ordered:
        for key in _entity_name_keys(item):
            name_blocks[key].add(item.entity_id)
    for key, members in sorted(name_blocks.items()):
        _reserve_block_pairs(tuple(members), reserved_pairs)
        for left_id, right_id in combinations(sorted(members), 2):
            pair = _pair(left_id, right_id)
            if pair in stable_edges:
                continue
            left, right = by_id[left_id], by_id[right_id]
            conflict = _pair_conflict(left, right)
            if conflict:
                rejected_decisions.setdefault(
                    pair,
                    _rejected(
                        left_id,
                        right_id,
                        method="exact_name_or_alias",
                        reason=conflict,
                        evidence_key=key,
                    ),
                )
            else:
                current = lower_edges.get(pair)
                candidate = _EvidenceEdge(
                    *pair,
                    method="exact_name_or_alias",
                    evidence_key=key,
                )
                if current is None or (key, pair) < (current.evidence_key, pair):
                    lower_edges[pair] = candidate

    acronym_blocks: dict[tuple[str, str], set[int]] = defaultdict(set)
    acronym_surfaces: dict[str, set[int]] = defaultdict(set)
    for item in ordered:
        bindings = _acronym_bindings(item)
        if _is_acronym(item.label):
            acronym_surfaces[_normalized_key(item.label)].add(item.entity_id)
        for acronym, full_forms in bindings.items():
            acronym_surfaces[acronym].add(item.entity_id)
            if len(full_forms) == 1:
                acronym_blocks[(acronym, next(iter(full_forms)))].add(item.entity_id)
    for (acronym, full_form), members in sorted(acronym_blocks.items()):
        _reserve_block_pairs(tuple(members), reserved_pairs)
        for left_id, right_id in combinations(sorted(members), 2):
            pair = _pair(left_id, right_id)
            if pair in stable_edges or pair in lower_edges:
                continue
            left, right = by_id[left_id], by_id[right_id]
            conflict = _pair_conflict(left, right)
            if conflict:
                rejected_decisions.setdefault(
                    pair,
                    _rejected(
                        left_id,
                        right_id,
                        method="defined_acronym",
                        reason=conflict,
                        evidence_key=f"{acronym}:{full_form}",
                    ),
                )
            else:
                lower_edges[pair] = _EvidenceEdge(
                    *pair,
                    method="defined_acronym",
                    evidence_key=f"{acronym}:{full_form}",
                )
    for acronym, members in sorted(acronym_surfaces.items()):
        _reserve_block_pairs(tuple(members), reserved_pairs)
        for left_id, right_id in combinations(sorted(members), 2):
            pair = _pair(left_id, right_id)
            if (
                pair in stable_edges
                or pair in lower_edges
                or pair in rejected_decisions
            ):
                continue
            left_forms = _acronym_bindings(by_id[left_id]).get(acronym, frozenset())
            right_forms = _acronym_bindings(by_id[right_id]).get(acronym, frozenset())
            reason = (
                "ambiguous_acronym"
                if len(left_forms) > 1 or len(right_forms) > 1
                else "undefined_acronym"
            )
            rejected_decisions[pair] = _rejected(
                left_id,
                right_id,
                method="defined_acronym",
                reason=reason,
                evidence_key=acronym,
            )

    dsu = _DisjointSet(by_id)
    accepted_edges: list[_EvidenceEdge] = []
    # Stable-ID blocks are preflighted as whole blocks, including a same-
    # collection cannot-link, before any union occurs.
    stable_by_identifier: dict[str, list[_EvidenceEdge]] = defaultdict(list)
    for edge in stable_edges.values():
        stable_by_identifier[edge.evidence_key].append(edge)
    for identifier, edges in sorted(stable_by_identifier.items()):
        member_ids = sorted(
            {value for edge in edges for value in (edge.left, edge.right)}
        )
        conflict = _component_conflict(member_ids, by_id)
        if conflict:
            for edge in edges:
                rejected_decisions[_pair(edge.left, edge.right)] = _rejected(
                    edge.left,
                    edge.right,
                    method=edge.method,
                    reason=conflict,
                    evidence_key=identifier,
                )
            continue
        for edge in sorted(edges, key=lambda item: (item.left, item.right)):
            dsu.union(edge.left, edge.right)
            accepted_edges.append(edge)

    # Examine the entire transitive lower-tier candidate component before any
    # union so an identifier-free bridge cannot attach arbitrarily to one of
    # two conflicting anchored identities or model versions.
    root_edges: list[_EvidenceEdge] = []
    for edge in sorted(
        lower_edges.values(),
        key=lambda item: (
            _METHOD_PRIORITY[item.method],
            item.evidence_key,
            item.left,
            item.right,
        ),
    ):
        left_root, right_root = dsu.find(edge.left), dsu.find(edge.right)
        if left_root != right_root:
            root_edges.append(
                _EvidenceEdge(
                    *_pair(left_root, right_root),
                    method=edge.method,
                    evidence_key=edge.evidence_key,
                )
            )
    root_to_members: dict[int, tuple[int, ...]] = {}
    for group in dsu.groups():
        root_to_members[dsu.find(group[0])] = group
    for root_group in _connected_edge_groups(root_to_members, root_edges):
        group_set = set(root_group)
        group_edges = tuple(
            edge
            for edge in root_edges
            if edge.left in group_set and edge.right in group_set
        )
        member_ids = tuple(
            sorted(
                value
                for root in root_group
                for value in root_to_members.get(root, (root,))
            )
        )
        conflict = _component_conflict(member_ids, by_id)
        if conflict:
            for original in lower_edges.values():
                endpoints_in_group = (
                    dsu.find(original.left) in group_set
                    and dsu.find(original.right) in group_set
                )
                if endpoints_in_group:
                    pair = _pair(original.left, original.right)
                    rejected_decisions[pair] = _rejected(
                        original.left,
                        original.right,
                        method=original.method,
                        reason=conflict,
                        evidence_key=original.evidence_key,
                    )
            continue
        for edge in group_edges:
            dsu.union(edge.left, edge.right)
        for original in lower_edges.values():
            if original.left in member_ids and original.right in member_ids:
                accepted_edges.append(original)

    decisions: dict[tuple[int, int], CanonicalDecision] = dict(rejected_decisions)
    for edge in accepted_edges:
        decisions[_pair(edge.left, edge.right)] = _automatic(edge)

    candidate_pairs = [
        _pair(item.left_entity_id, item.right_entity_id)
        for item in embedding_candidates
    ]
    if len(candidate_pairs) != len(set(candidate_pairs)):
        raise ValueError("duplicate embedding candidate pair")
    for candidate in sorted(
        embedding_candidates,
        key=lambda item: (item.left_entity_id, item.right_entity_id),
    ):
        pair = _pair(candidate.left_entity_id, candidate.right_entity_id)
        if pair[0] not in by_id or pair[1] not in by_id:
            raise ValueError(
                "embedding candidate endpoint is outside the entity snapshot"
            )
        if pair in decisions or dsu.find(pair[0]) == dsu.find(pair[1]):
            continue
        left, right = by_id[pair[0]], by_id[pair[1]]
        conflict = _pair_conflict(left, right)
        if (
            conflict is None
            and left.identifier
            and right.identifier
            and left.identifier != right.identifier
        ):
            conflict = "conflicting_stable_identifiers"
        if conflict is None and _is_acronym(left.label) and _is_acronym(right.label):
            left_key = _normalized_key(left.label)
            right_key = _normalized_key(right.label)
            left_forms = _acronym_bindings(left).get(left_key, frozenset())
            right_forms = _acronym_bindings(right).get(right_key, frozenset())
            if left_key == right_key and (
                len(left_forms) != 1
                or len(right_forms) != 1
                or left_forms != right_forms
            ):
                conflict = "ambiguous_acronym"
        if conflict:
            decisions[pair] = _rejected(
                *pair,
                method="embedding_similarity",
                reason=conflict,
                evidence_key=candidate.embedding_model_signature,
            )
            continue
        decisions[pair] = CanonicalDecision(
            *pair,
            score=candidate.similarity,
            method="embedding_similarity",
            outcome=CanonicalOutcome.CANDIDATE,
            reason="embedding_similarity_requires_review",
            evidence_key=candidate.embedding_model_signature,
            metadata=(
                ("embedding_model_signature", candidate.embedding_model_signature),
                ("left_input_hash", candidate.left_input_hash),
                ("right_input_hash", candidate.right_input_hash),
            ),
        )
        if len(decisions) > MAX_CANONICAL_DECISIONS:
            raise ValueError("canonical decision cap exceeded")

    accepted_tuple = tuple(accepted_edges)
    components: list[CanonicalComponent] = []
    for group in dsu.groups():
        members = tuple(by_id[value] for value in group)
        representative = min(
            members,
            key=lambda item: (
                item.normalized_label,
                item.label,
                item.collection_id,
                item.entity_id,
            ),
        )
        component_edges = tuple(
            edge
            for edge in accepted_tuple
            if edge.left in group and edge.right in group
        )
        payload = _component_identity_payload(members, component_edges)
        method = (
            "singleton"
            if len(group) == 1
            else min(
                component_edges,
                key=lambda edge: _METHOD_PRIORITY[edge.method],
            ).method
        )
        components.append(
            CanonicalComponent(
                identity_key=_hash_payload(
                    {"resolver_version": resolver_version, "identity": payload}
                ),
                entity_ids=group,
                collection_ids=tuple(sorted(item.collection_id for item in members)),
                label=representative.label,
                normalized_label=representative.normalized_label,
                entity_type=representative.entity_type,
                version_signature=representative.version_signature,
                method=method,
            )
        )
    ordered_components = tuple(sorted(components, key=lambda item: item.entity_ids))
    ordered_decisions = tuple(
        sorted(
            decisions.values(),
            key=lambda item: (
                item.left_entity_id,
                item.right_entity_id,
                _METHOD_PRIORITY[item.method],
                item.outcome.value,
            ),
        )
    )
    checksum = _hash_payload(
        {
            "resolver_version": resolver_version,
            "components": [
                {
                    "identity_key": item.identity_key,
                    "entity_ids": list(item.entity_ids),
                    "collection_ids": list(item.collection_ids),
                    "label": item.label,
                    "normalized_label": item.normalized_label,
                    "entity_type": item.entity_type,
                    "version_signature": item.version_signature,
                    "method": item.method,
                }
                for item in ordered_components
            ],
            "decisions": [
                {
                    "left": item.left_entity_id,
                    "right": item.right_entity_id,
                    "score": item.score,
                    "method": item.method,
                    "outcome": item.outcome.value,
                    "reason": item.reason,
                    "evidence_key": item.evidence_key,
                    "metadata": list(item.metadata),
                }
                for item in ordered_decisions
            ],
        }
    )
    return CanonicalResolutionResult(
        resolver_version=resolver_version,
        components=ordered_components,
        decisions=ordered_decisions,
        checksum=checksum,
    )


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _projected_decision_audit(decision: CanonicalDecision) -> dict[str, object]:
    """Retain evidence without persisting another collection's entity ID/label."""

    metadata = dict(decision.metadata)
    return {
        "decision_checksum": _hash_payload(
            {
                "left_entity_id": decision.left_entity_id,
                "right_entity_id": decision.right_entity_id,
                "score": decision.score,
                "method": decision.method,
                "outcome": decision.outcome.value,
                "reason": decision.reason,
                "evidence_key": decision.evidence_key,
                "metadata": metadata,
            }
        ),
        "evidence_key_checksum": _hash_payload(decision.evidence_key),
        "metadata": metadata,
        "method": decision.method,
        "outcome": decision.outcome.value,
        "reason": decision.reason,
        "score": decision.score,
    }


def _decision_order(decision: CanonicalDecision) -> tuple[object, ...]:
    return (
        _METHOD_PRIORITY[decision.method],
        -decision.score,
        decision.reason,
        decision.evidence_key,
        decision.left_entity_id,
        decision.right_entity_id,
    )


def project_canonical_link_decisions(
    resolution: CanonicalResolutionResult,
) -> tuple[CanonicalProjectedLinkDecision, ...]:
    """Project pair decisions into unique, privacy-safe source registry audits."""

    if type(resolution) is not CanonicalResolutionResult:
        raise ValueError("resolution must be an exact CanonicalResolutionResult")
    component_by_entity: dict[int, CanonicalComponent] = {}
    for component in resolution.components:
        for entity_id in component.entity_ids:
            if entity_id in component_by_entity:
                raise ValueError("entity appears in multiple canonical components")
            component_by_entity[entity_id] = component

    automatic_by_source: dict[int, list[CanonicalDecision]] = defaultdict(list)
    folded_by_source: dict[int, list[CanonicalDecision]] = defaultdict(list)
    cross_groups: dict[tuple[int, str], list[CanonicalDecision]] = defaultdict(list)
    for decision in resolution.decisions:
        left_component = component_by_entity.get(decision.left_entity_id)
        right_component = component_by_entity.get(decision.right_entity_id)
        if left_component is None or right_component is None:
            raise ValueError("decision endpoint escaped canonical components")
        if decision.outcome is CanonicalOutcome.AUTOMATIC:
            if left_component.identity_key != right_component.identity_key:
                raise ValueError("automatic decision crosses canonical components")
            automatic_by_source[decision.left_entity_id].append(decision)
            automatic_by_source[decision.right_entity_id].append(decision)
            continue

        target_component = right_component
        incompatible_target = (
            left_component.entity_type != right_component.entity_type
            or left_component.version_signature != right_component.version_signature
        )
        if (
            left_component.identity_key == right_component.identity_key
            or incompatible_target
        ):
            target_component = left_component
            if len(left_component.entity_ids) >= 2:
                folded_by_source[decision.left_entity_id].append(decision)
                continue
        cross_groups[(decision.left_entity_id, target_component.identity_key)].append(
            decision
        )

    projected: list[CanonicalProjectedLinkDecision] = []
    for component in resolution.components:
        if len(component.entity_ids) < 2:
            continue
        for source_entity_id in component.entity_ids:
            evidence = tuple(
                sorted(automatic_by_source[source_entity_id], key=_decision_order)
            )
            if not evidence:
                raise ValueError("automatic component member has no accepted evidence")
            representative = evidence[0]
            folded = tuple(
                sorted(folded_by_source[source_entity_id], key=_decision_order)
            )
            projected.append(
                CanonicalProjectedLinkDecision(
                    source_entity_id=source_entity_id,
                    target_identity_key=component.identity_key,
                    score=representative.score,
                    method=representative.method,
                    outcome=CanonicalOutcome.AUTOMATIC,
                    status="active",
                    reason=representative.reason,
                    metadata_json=_canonical_json(
                        {
                            "component_identity_key": component.identity_key,
                            "folded_decisions": [
                                _projected_decision_audit(item) for item in folded
                            ],
                            "source_evidence": [
                                _projected_decision_audit(item) for item in evidence
                            ],
                        }
                    ),
                )
            )

    for (source_entity_id, target_identity_key), decisions in sorted(
        cross_groups.items()
    ):
        ordered = tuple(sorted(decisions, key=_decision_order))
        if any(item.outcome is CanonicalOutcome.REJECTED for item in ordered):
            outcome = CanonicalOutcome.REJECTED
            representative = next(
                item for item in ordered if item.outcome is CanonicalOutcome.REJECTED
            )
            status = "rejected"
        else:
            if any(item.outcome is not CanonicalOutcome.CANDIDATE for item in ordered):
                raise ValueError("cross-component audit has an invalid outcome")
            outcome = CanonicalOutcome.CANDIDATE
            representative = min(
                ordered,
                key=lambda item: (
                    -item.score,
                    item.evidence_key,
                    item.left_entity_id,
                    item.right_entity_id,
                ),
            )
            status = "suppressed"
        projected.append(
            CanonicalProjectedLinkDecision(
                source_entity_id=source_entity_id,
                target_identity_key=target_identity_key,
                score=representative.score,
                method=representative.method,
                outcome=outcome,
                status=status,
                reason=representative.reason,
                metadata_json=_canonical_json(
                    {
                        "pair_decisions": [
                            _projected_decision_audit(item) for item in ordered
                        ],
                        "target_identity_key": target_identity_key,
                    }
                ),
            )
        )
    if len(projected) > MAX_CANONICAL_MEMBERSHIPS:
        raise ValueError("projected canonical links exceed the membership cap")
    result = tuple(
        sorted(
            projected,
            key=lambda item: (
                item.source_entity_id,
                item.target_identity_key,
                item.outcome.value,
                item.method,
            ),
        )
    )
    keys = {(item.source_entity_id, item.target_identity_key) for item in result}
    if len(keys) != len(result):
        raise ValueError("projected canonical link key is not unique")
    return result


@dataclass(frozen=True, slots=True)
class PermissionBearingEntity:
    collection_entity_id: int
    collection_id: int
    artifact_id: int
    document_ids: tuple[uuid.UUID, ...]

    def __post_init__(self) -> None:
        _positive_int(self.collection_entity_id, "collection_entity_id")
        _positive_int(self.collection_id, "collection_id")
        _positive_int(self.artifact_id, "artifact_id")
        if type(self.document_ids) is not tuple or any(
            type(item) is not uuid.UUID for item in self.document_ids
        ):
            raise ValueError("document_ids must be an exact UUID tuple")
        if self.document_ids != tuple(sorted(set(self.document_ids), key=str)):
            raise ValueError("document_ids must be canonically sorted and unique")


@dataclass(frozen=True, slots=True)
class CanonicalMembershipRow:
    canonical_entity_id: int
    collection_entity_id: int
    collection_id: int
    artifact_id: int

    def __post_init__(self) -> None:
        _positive_int(self.canonical_entity_id, "canonical_entity_id")
        _positive_int(self.collection_entity_id, "collection_entity_id")
        _positive_int(self.collection_id, "collection_id")
        _positive_int(self.artifact_id, "artifact_id")


@dataclass(frozen=True, slots=True)
class CanonicalLookupResult:
    canonical_by_seed: tuple[tuple[int, int], ...]
    members_by_canonical: tuple[tuple[int, tuple[tuple[int, int, int], ...]], ...]

    def __post_init__(self) -> None:
        if (
            type(self.canonical_by_seed) is not tuple
            or type(self.members_by_canonical) is not tuple
        ):
            raise ValueError("canonical lookup rows must be exact tuples")
        if self.canonical_by_seed != tuple(sorted(set(self.canonical_by_seed))):
            raise ValueError("canonical seed mappings must be sorted and unique")
        canonical_ids = tuple(item[0] for item in self.members_by_canonical)
        if canonical_ids != tuple(sorted(set(canonical_ids))):
            raise ValueError("canonical member groups must be sorted and unique")


def _sorted_positive_tuple(
    values: object, label: str, *, maximum: int
) -> tuple[int, ...]:
    if type(values) is not tuple or any(type(value) is not int for value in values):
        raise ValueError(f"{label} must be an exact integer tuple")
    if len(values) > maximum or any(not 1 <= value <= 2**63 - 1 for value in values):
        raise ValueError(f"{label} exceeds its positive integer envelope")
    if values != tuple(sorted(set(values))):
        raise ValueError(f"{label} must be canonically sorted and unique")
    return values


def _sorted_uuid_tuple(values: object, label: str) -> tuple[uuid.UUID, ...]:
    if type(values) is not tuple or any(
        type(value) is not uuid.UUID for value in values
    ):
        raise ValueError(f"{label} must be an exact UUID tuple")
    if len(values) > MAX_CANONICAL_ENTITIES:
        raise ValueError(f"{label} exceeds its UUID envelope")
    if values != tuple(sorted(set(values), key=str)):
        raise ValueError(f"{label} must be canonically sorted and unique")
    return values


def _batches(values: Sequence[object]) -> Iterable[tuple[object, ...]]:
    for start in range(0, len(values), CANONICAL_QUERY_BATCH_SIZE):
        yield tuple(values[start : start + CANONICAL_QUERY_BATCH_SIZE])


def _bounded_extend(
    target: list[object], rows: Iterable[object], *, maximum: int, label: str
) -> None:
    for row in rows:
        target.append(row)
        if len(target) > maximum:
            raise ValueError(f"{label} exceeds the canonical row cap")


def project_authorized_canonical_lookup(
    *,
    seed_collection_entity_ids: tuple[int, ...],
    permission_endpoints: tuple[PermissionBearingEntity, ...],
    canonical_memberships: tuple[CanonicalMembershipRow, ...],
    allowed_collection_ids: tuple[int, ...],
    allowed_document_ids: tuple[uuid.UUID, ...],
    active_artifact_ids: tuple[int, ...],
) -> CanonicalLookupResult:
    """Project only memberships whose local endpoint is independently allowed."""

    seeds = _sorted_positive_tuple(
        seed_collection_entity_ids,
        "seed_collection_entity_ids",
        maximum=MAX_CANONICAL_ENTITIES,
    )
    collections = _sorted_positive_tuple(
        allowed_collection_ids,
        "allowed_collection_ids",
        maximum=MAX_CANONICAL_COLLECTIONS,
    )
    documents = _sorted_uuid_tuple(allowed_document_ids, "allowed_document_ids")
    active_artifacts = _sorted_positive_tuple(
        active_artifact_ids,
        "active_artifact_ids",
        maximum=MAX_CANONICAL_ARTIFACTS,
    )
    if type(permission_endpoints) is not tuple or any(
        type(item) is not PermissionBearingEntity for item in permission_endpoints
    ):
        raise ValueError("permission_endpoints must be an exact typed tuple")
    if len(permission_endpoints) > MAX_CANONICAL_SOURCE_LINKS:
        raise ValueError("permission endpoints exceed the source-link cap")
    if type(canonical_memberships) is not tuple or any(
        type(item) is not CanonicalMembershipRow for item in canonical_memberships
    ):
        raise ValueError("canonical_memberships must be an exact typed tuple")
    if len(canonical_memberships) > MAX_CANONICAL_ENTITIES:
        raise ValueError("canonical memberships exceed the entity cap")
    endpoint_ids = [item.collection_entity_id for item in permission_endpoints]
    if len(endpoint_ids) != len(set(endpoint_ids)):
        raise ValueError("permission endpoints contain duplicate entity IDs")
    membership_ids = [item.collection_entity_id for item in canonical_memberships]
    if len(membership_ids) != len(set(membership_ids)):
        raise ValueError("canonical memberships contain duplicate source entities")
    allowed_collections = frozenset(collections)
    allowed_documents = frozenset(documents)
    allowed_artifacts = frozenset(active_artifacts)
    authorized_entities = {
        item.collection_entity_id: item
        for item in permission_endpoints
        if item.collection_id in allowed_collections
        and item.artifact_id in allowed_artifacts
        and bool(allowed_documents.intersection(item.document_ids))
    }
    memberships = tuple(
        sorted(
            (
                item
                for item in canonical_memberships
                if item.collection_entity_id in authorized_entities
                and item.collection_id
                == authorized_entities[item.collection_entity_id].collection_id
                and item.artifact_id
                == authorized_entities[item.collection_entity_id].artifact_id
            ),
            key=lambda item: (
                item.canonical_entity_id,
                item.collection_id,
                item.collection_entity_id,
                item.artifact_id,
            ),
        )
    )
    canonical_by_entity = {
        item.collection_entity_id: item.canonical_entity_id for item in memberships
    }
    canonical_by_seed = tuple(
        (seed, canonical_by_entity[seed])
        for seed in seeds
        if seed in authorized_entities and seed in canonical_by_entity
    )
    selected_canonical_ids = {canonical_id for _, canonical_id in canonical_by_seed}
    grouped: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for item in memberships:
        if item.canonical_entity_id in selected_canonical_ids:
            grouped[item.canonical_entity_id].append(
                (item.collection_id, item.collection_entity_id, item.artifact_id)
            )
    return CanonicalLookupResult(
        canonical_by_seed=canonical_by_seed,
        members_by_canonical=tuple(
            (canonical_id, tuple(sorted(set(values))))
            for canonical_id, values in sorted(grouped.items())
        ),
    )


def authorized_canonical_lookup(
    *,
    seed_collection_entity_ids: tuple[int, ...],
    allowed_collection_ids: tuple[int, ...],
    allowed_document_ids: tuple[uuid.UUID, ...],
    active_artifact_ids: tuple[int, ...],
    resolver_version: str,
) -> CanonicalLookupResult:
    """Load one exact DB-derived, permission-first canonical projection."""

    from django.db.models import F

    from apps.knowledge_graph.models import (
        CanonicalEntityLink,
        CollectionEntityDocumentLink,
    )

    seeds = _sorted_positive_tuple(
        seed_collection_entity_ids,
        "seed_collection_entity_ids",
        maximum=MAX_CANONICAL_ENTITIES,
    )
    collections = _sorted_positive_tuple(
        allowed_collection_ids,
        "allowed_collection_ids",
        maximum=MAX_CANONICAL_COLLECTIONS,
    )
    documents = _sorted_uuid_tuple(allowed_document_ids, "allowed_document_ids")
    artifacts = _sorted_positive_tuple(
        active_artifact_ids,
        "active_artifact_ids",
        maximum=MAX_CANONICAL_ARTIFACTS,
    )
    _bounded_text(resolver_version, "resolver_version", maximum=128)
    if not seeds or not collections or not documents or not artifacts:
        return CanonicalLookupResult((), ())

    # Establish every permission-bearing local endpoint before reading a
    # canonical link.  Each batch repeats the complete active artifact,
    # collection, and document predicates; discovered IDs never authorize.
    permission_rows: list[object] = []
    for document_batch in _batches(documents):
        query = (
            CollectionEntityDocumentLink.objects.current()
            .filter(
                artifact_id__in=artifacts,
                collection_entity__artifact_id__in=artifacts,
                collection_entity__artifact_id=F("artifact_id"),
                manifest_input__artifact_id=F("artifact_id"),
                collection_entity__collection_id__in=collections,
                manifest_input__collection_id__in=collections,
                manifest_input__collection_id=F("collection_entity__collection_id"),
                manifest_input__document_id__in=document_batch,
                document_entity__document_id__in=document_batch,
                manifest_input__document_id=F("document_entity__document_id"),
                manifest_input__document_artifact_id=F("document_entity__artifact_id"),
            )
            .order_by("collection_entity_id", "document_entity__document_id", "pk")
            .values_list(
                "collection_entity_id",
                "collection_entity__collection_id",
                "collection_entity__artifact_id",
                "document_entity__document_id",
            )
        )
        _bounded_extend(
            permission_rows,
            query.iterator(chunk_size=1_000),
            maximum=MAX_CANONICAL_SOURCE_LINKS,
            label="permission-bearing endpoint",
        )

    endpoint_values: dict[int, tuple[int, int, set[uuid.UUID]]] = {}
    for entity_id, collection_id, artifact_id, document_id in permission_rows:
        key = _positive_int(entity_id, "permission collection entity id")
        scope = (
            _positive_int(collection_id, "permission collection id"),
            _positive_int(artifact_id, "permission artifact id"),
        )
        existing = endpoint_values.get(key)
        if existing is not None and existing[:2] != scope:
            raise ValueError("permission endpoint has conflicting active scope")
        if type(document_id) is not uuid.UUID:
            raise ValueError("permission endpoint document is not an exact UUID")
        if existing is None:
            endpoint_values[key] = (scope[0], scope[1], {document_id})
        else:
            existing[2].add(document_id)
    endpoints = tuple(
        PermissionBearingEntity(
            collection_entity_id=entity_id,
            collection_id=collection_id,
            artifact_id=artifact_id,
            document_ids=tuple(sorted(document_ids, key=str)),
        )
        for entity_id, (collection_id, artifact_id, document_ids) in sorted(
            endpoint_values.items()
        )
    )
    authorized_entity_ids = tuple(item.collection_entity_id for item in endpoints)
    authorized_set = frozenset(authorized_entity_ids)
    authorized_seeds = tuple(seed for seed in seeds if seed in authorized_set)
    if not authorized_seeds:
        return CanonicalLookupResult((), ())

    seed_link_rows: list[object] = []
    for seed_batch in _batches(authorized_seeds):
        query = (
            CanonicalEntityLink.objects.current(resolver_version=resolver_version)
            .filter(
                collection_entity_id__in=seed_batch,
                collection_entity__artifact_id__in=artifacts,
                collection_entity__collection_id__in=collections,
            )
            .order_by("collection_entity_id", "canonical_entity_id", "pk")
            .values_list("collection_entity_id", "canonical_entity_id")
        )
        _bounded_extend(
            seed_link_rows,
            query.iterator(chunk_size=1_000),
            maximum=MAX_CANONICAL_ENTITIES,
            label="authorized seed canonical link",
        )
    canonical_ids_by_seed: dict[int, int] = {}
    for entity_id, canonical_id in seed_link_rows:
        entity_id = _positive_int(entity_id, "canonical seed entity id")
        canonical_id = _positive_int(canonical_id, "canonical seed id")
        existing = canonical_ids_by_seed.setdefault(entity_id, canonical_id)
        if existing != canonical_id:
            raise ValueError("authorized seed has multiple current canonical targets")
    selected_canonical_ids = tuple(sorted(set(canonical_ids_by_seed.values())))
    if not selected_canonical_ids:
        return CanonicalLookupResult((), ())

    membership_values: list[object] = []
    for canonical_batch in _batches(selected_canonical_ids):
        for entity_batch in _batches(authorized_entity_ids):
            query = (
                CanonicalEntityLink.objects.current(resolver_version=resolver_version)
                .filter(
                    canonical_entity_id__in=canonical_batch,
                    collection_entity_id__in=entity_batch,
                    collection_entity__artifact_id__in=artifacts,
                    collection_entity__collection_id__in=collections,
                )
                .order_by(
                    "canonical_entity_id",
                    "collection_entity__collection_id",
                    "collection_entity_id",
                    "pk",
                )
                .values_list(
                    "canonical_entity_id",
                    "collection_entity_id",
                    "collection_entity__collection_id",
                    "collection_entity__artifact_id",
                )
            )
            _bounded_extend(
                membership_values,
                query.iterator(chunk_size=1_000),
                maximum=MAX_CANONICAL_ENTITIES,
                label="authorized canonical membership",
            )
    memberships = tuple(
        CanonicalMembershipRow(*row) for row in sorted(set(membership_values))
    )
    return project_authorized_canonical_lookup(
        seed_collection_entity_ids=seeds,
        permission_endpoints=endpoints,
        canonical_memberships=memberships,
        allowed_collection_ids=collections,
        allowed_document_ids=documents,
        active_artifact_ids=artifacts,
    )


class _CanonicalSnapshotChanged(RuntimeError):
    """Retryable drift while acquiring the complete pre-registry lock prefix."""


_CANONICAL_REGISTRY_ADVISORY_LOCK = 5_497_240_000_000_000_000
_CANONICAL_REBUILD_ATTEMPTS = 3


def _active_collection_artifact_snapshot(*, using: str) -> tuple[tuple[int, int], ...]:
    from apps.knowledge_graph.models import GraphArtifact

    rows = list(
        GraphArtifact.objects.using(using)
        .filter(
            scope_type=GraphArtifact.ScopeType.COLLECTION,
            status=GraphArtifact.Status.ACTIVE,
        )
        .order_by("collection_scope_id", "pk")
        .values_list("pk", "collection_scope_id")[: MAX_CANONICAL_ARTIFACTS + 1]
    )
    if len(rows) > MAX_CANONICAL_ARTIFACTS:
        raise ValueError("active collection artifacts exceed the canonical cap")
    snapshot: list[tuple[int, int]] = []
    collection_ids: set[int] = set()
    for artifact_id, collection_id in rows:
        artifact_id = _positive_int(artifact_id, "active artifact id")
        collection_id = _positive_int(collection_id, "active collection id")
        if collection_id in collection_ids:
            raise RuntimeError("collection has multiple active graph artifacts")
        collection_ids.add(collection_id)
        snapshot.append((artifact_id, collection_id))
    return tuple(snapshot)


def _bounded_locked_rows(
    rows: Iterable[object], *, maximum: int, label: str
) -> tuple[object, ...]:
    values: list[object] = []
    _bounded_extend(values, rows, maximum=maximum, label=label)
    return tuple(values)


def _load_locked_canonical_inputs(
    *,
    entity_rows: tuple[object, ...],
    active_artifact_ids: tuple[int, ...],
    using: str,
) -> tuple[CanonicalEntityInput, ...]:
    """Lock and adapt exact Task-8/9 provenance for the active C snapshot."""

    from django.db.models import F

    from apps.knowledge_graph.models import (
        CollectionEntityDocumentLink,
        DocumentEntity,
        DocumentEntityMention,
        GraphArtifact,
    )

    if not entity_rows:
        return ()
    entity_ids = tuple(row.pk for row in entity_rows)
    source_rows: list[object] = []
    for entity_batch in _batches(entity_ids):
        _bounded_extend(
            source_rows,
            CollectionEntityDocumentLink.objects.using(using)
            .select_for_update(of=("self",))
            .current()
            .filter(
                artifact_id__in=active_artifact_ids,
                collection_entity_id__in=entity_batch,
                collection_entity__artifact_id=F("artifact_id"),
                manifest_input__artifact_id=F("artifact_id"),
                manifest_input__collection_id=F("collection_entity__collection_id"),
                manifest_input__document_artifact_id=F("document_entity__artifact_id"),
                manifest_input__document_id=F("document_entity__document_id"),
            )
            .order_by("pk")
            .values_list(
                "collection_entity_id",
                "document_entity_id",
                "document_entity__artifact_id",
                "document_entity__document_id",
                "pk",
            )
            .iterator(chunk_size=1_000),
            maximum=MAX_CANONICAL_SOURCE_LINKS,
            label="canonical source memberships",
        )
    membership_keys: set[tuple[int, int, int]] = set()
    document_by_artifact: dict[int, uuid.UUID] = {}
    document_entity_scope: dict[int, tuple[int, uuid.UUID]] = {}
    source_entity_by_document_entity: dict[int, int] = {}
    for (
        collection_entity_id,
        document_entity_id,
        document_artifact_id,
        document_id,
        _link_id,
    ) in source_rows:
        collection_entity_id = _positive_int(
            collection_entity_id, "source collection entity id"
        )
        document_entity_id = _positive_int(
            document_entity_id, "source document entity id"
        )
        document_artifact_id = _positive_int(
            document_artifact_id, "source document artifact id"
        )
        if type(document_id) is not uuid.UUID:
            raise RuntimeError("source membership has a non-UUID document")
        key = (
            collection_entity_id,
            document_entity_id,
            document_artifact_id,
        )
        if key in membership_keys:
            raise RuntimeError("duplicate current canonical source membership")
        membership_keys.add(key)
        previous_document = document_by_artifact.setdefault(
            document_artifact_id, document_id
        )
        if previous_document != document_id:
            raise RuntimeError("document artifact has conflicting document UUIDs")
        previous_scope = document_entity_scope.setdefault(
            document_entity_id, (document_artifact_id, document_id)
        )
        if previous_scope != (document_artifact_id, document_id):
            raise RuntimeError("document entity has conflicting active source scope")
        previous_source = source_entity_by_document_entity.setdefault(
            document_entity_id, collection_entity_id
        )
        if previous_source != collection_entity_id:
            raise RuntimeError(
                "document entity has multiple current collection assignments"
            )

    document_artifact_ids = tuple(sorted(document_by_artifact))
    locked_document_artifacts: list[object] = []
    for artifact_batch in _batches(document_artifact_ids):
        _bounded_extend(
            locked_document_artifacts,
            GraphArtifact.objects.using(using)
            .select_for_update()
            .filter(pk__in=artifact_batch)
            .order_by("pk")
            .iterator(chunk_size=1_000),
            maximum=MAX_CANONICAL_SOURCE_LINKS,
            label="canonical document artifacts",
        )
    if tuple(row.pk for row in locked_document_artifacts) != document_artifact_ids:
        raise _CanonicalSnapshotChanged("canonical source artifact set changed")
    for artifact in locked_document_artifacts:
        if (
            artifact.scope_type != GraphArtifact.ScopeType.DOCUMENT
            or artifact.status != GraphArtifact.Status.ACTIVE
            or artifact.scope_id != str(document_by_artifact[artifact.pk])
        ):
            raise _CanonicalSnapshotChanged(
                "canonical source artifact is no longer exact and active"
            )

    document_entity_ids = tuple(sorted(document_entity_scope))
    locked_document_entities: list[object] = []
    for entity_batch in _batches(document_entity_ids):
        _bounded_extend(
            locked_document_entities,
            DocumentEntity.objects.using(using)
            .select_for_update()
            .filter(pk__in=entity_batch)
            .order_by("pk")
            .iterator(chunk_size=1_000),
            maximum=MAX_CANONICAL_SOURCE_LINKS,
            label="canonical document entities",
        )
    if tuple(row.pk for row in locked_document_entities) != document_entity_ids:
        raise _CanonicalSnapshotChanged("canonical document entity set changed")
    for document_entity in locked_document_entities:
        expected_artifact_id, expected_document_id = document_entity_scope[
            document_entity.pk
        ]
        if (
            document_entity.status != DocumentEntity.Status.ACTIVE
            or document_entity.artifact_id != expected_artifact_id
            or document_entity.document_id != expected_document_id
        ):
            raise _CanonicalSnapshotChanged(
                "canonical document entity is no longer exact and active"
            )

    mention_rows: list[object] = []
    for entity_batch in _batches(document_entity_ids):
        _bounded_extend(
            mention_rows,
            DocumentEntityMention.objects.using(using)
            .select_for_update(of=("self",))
            .filter(
                document_entity_id__in=entity_batch,
                status=DocumentEntityMention.Status.ACTIVE,
                document_entity__status=DocumentEntity.Status.ACTIVE,
                document_entity__artifact__status=GraphArtifact.Status.ACTIVE,
                mention__artifact__status=GraphArtifact.Status.ACTIVE,
                resolver_version=F("document_entity__artifact__resolver_version"),
                mention__artifact_id=F("document_entity__artifact_id"),
                mention__document_id=F("document_entity__document_id"),
            )
            .select_related("mention")
            .order_by("pk")
            .iterator(chunk_size=1_000),
            maximum=MAX_CANONICAL_MEMBERSHIPS,
            label="canonical mention provenance",
        )
    mention_by_key: dict[tuple[int, int], object] = {}
    for row in mention_rows:
        key = (row.document_entity_id, row.mention_id)
        if key in mention_by_key:
            raise RuntimeError("duplicate active mention membership")
        mention_by_key[key] = row

    provenance: list[CanonicalProvenanceRow] = []
    for row in mention_rows:
        if row.method not in {"ontology_alias", "defined_acronym"}:
            continue
        parent_value = row.parent_mention_id
        if type(parent_value) is not str or not re.fullmatch(
            r"[1-9][0-9]*", parent_value
        ):
            raise RuntimeError("eligible canonical provenance has an invalid parent")
        parent_id = int(parent_value)
        _positive_int(parent_id, "canonical parent mention id")
        parent = mention_by_key.get((row.document_entity_id, parent_id))
        if parent is None:
            raise RuntimeError(
                "eligible canonical provenance escaped its active document cluster"
            )
        collection_entity_id = source_entity_by_document_entity[row.document_entity_id]
        document_artifact_id, _document_id = document_entity_scope[
            row.document_entity_id
        ]
        provenance.append(
            CanonicalProvenanceRow(
                collection_entity_id=collection_entity_id,
                document_entity_id=row.document_entity_id,
                document_artifact_id=document_artifact_id,
                mention_id=row.mention_id,
                parent_mention_id=parent_id,
                method=row.method,
                surface=row.mention.raw_text,
                parent_surface=parent.mention.raw_text,
                source_collection_entity_id=collection_entity_id,
            )
        )
        if len(provenance) > MAX_CANONICAL_MEMBERSHIPS:
            raise ValueError("canonical provenance exceeds the membership cap")

    memberships = tuple(
        CanonicalSourceMembership(*key) for key in sorted(membership_keys)
    )
    return build_canonical_inputs_from_provenance(
        entity_rows,
        tuple(
            sorted(
                provenance,
                key=lambda row: (
                    row.collection_entity_id,
                    row.document_entity_id,
                    row.mention_id,
                    row.parent_mention_id,
                ),
            )
        ),
        memberships,
    )


def _lock_canonical_registry(*, using: str) -> None:
    from django.db import connections

    connection = connections[using]
    if connection.vendor != "postgresql":
        return
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT pg_advisory_xact_lock(%s)",
            [_CANONICAL_REGISTRY_ADVISORY_LOCK],
        )


def _derive_locked_embedding_candidates(
    entity_rows: tuple[object, ...],
    *,
    artifact_embedding_signatures: Mapping[int, str],
) -> tuple[CanonicalEmbeddingCandidate, ...]:
    """Build a bounded deterministic review set from the exact locked snapshot."""

    import numpy as np

    eligible: list[tuple[object, np.ndarray]] = []
    for row in entity_rows:
        expected_signature = artifact_embedding_signatures.get(row.artifact_id)
        if expected_signature is None:
            raise RuntimeError(
                "embedding endpoint escaped the locked artifact snapshot"
            )
        if row.embedding is None:
            if row.embedding_model_signature or row.embedding_input_hash:
                raise RuntimeError("embedding audit exists without a source vector")
            continue
        if (
            type(row.embedding_model_signature) is not str
            or not row.embedding_model_signature
            or row.embedding_model_signature != expected_signature
            or type(row.embedding_input_hash) is not str
            or not _HASH_PATTERN.fullmatch(row.embedding_input_hash)
        ):
            raise RuntimeError("locked embedding audit is incomplete")
        try:
            vector = np.asarray(tuple(row.embedding), dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError("locked embedding vector is invalid") from exc
        if vector.shape != (1024,) or not np.isfinite(vector).all():
            raise RuntimeError("locked embedding vector is not 1024 finite values")
        norm = float(np.linalg.norm(vector))
        if not isfinite(norm) or norm <= 0:
            continue
        eligible.append((row, vector / norm))

    # A fixed signed projection gives an input-order-independent locality
    # window.  At most four later neighbors are compared for each entity, so
    # the intermediate pair set is O(N) and remains below the decision cap.
    weights = np.fromiter(
        (
            1.0 if (((index * 1_103_515_245 + 12_345) >> 16) & 1) else -1.0
            for index in range(1024)
        ),
        dtype=np.float64,
        count=1024,
    )
    grouped: dict[tuple[str, str, str], list[tuple[object, np.ndarray, float]]] = (
        defaultdict(list)
    )
    for row, vector in eligible:
        projection = float(np.dot(vector, weights))
        grouped[
            (
                row.embedding_model_signature,
                row.entity_type,
                row.version_signature,
            )
        ].append((row, vector, projection))

    candidates: dict[tuple[int, int], CanonicalEmbeddingCandidate] = {}
    considered_pairs = 0
    for group_key, values in sorted(grouped.items()):
        ordered = tuple(sorted(values, key=lambda item: (item[2], item[0].pk)))
        for index, (left, left_vector, _projection) in enumerate(ordered):
            stop = min(len(ordered), index + CANONICAL_EMBEDDING_PROJECTION_WINDOW + 1)
            for right, right_vector, _right_projection in ordered[index + 1 : stop]:
                considered_pairs += 1
                if considered_pairs > MAX_CANONICAL_DECISIONS:
                    raise ValueError("embedding candidate pair cap exceeded")
                if left.collection_id == right.collection_id:
                    continue
                similarity = float(np.dot(left_vector, right_vector))
                if (
                    not isfinite(similarity)
                    or similarity < CANONICAL_EMBEDDING_CANDIDATE_MIN_SIMILARITY
                ):
                    continue
                similarity = min(1.0, max(0.0, round(similarity, 12)))
                candidate = CanonicalEmbeddingCandidate(
                    left_entity_id=left.pk,
                    right_entity_id=right.pk,
                    similarity=similarity,
                    embedding_model_signature=group_key[0],
                    left_input_hash=left.embedding_input_hash,
                    right_input_hash=right.embedding_input_hash,
                )
                key = (candidate.left_entity_id, candidate.right_entity_id)
                previous = candidates.get(key)
                if previous is not None and previous != candidate:
                    raise RuntimeError("embedding candidate audit is not deterministic")
                candidates[key] = candidate
    if len(candidates) > MAX_CANONICAL_DECISIONS:
        raise ValueError("embedding candidate decision cap exceeded")
    return tuple(candidates[key] for key in sorted(candidates))


def _reconcile_locked_registry(
    *,
    resolution: CanonicalResolutionResult,
    active_artifact_ids: tuple[int, ...],
    source_entities_by_id: Mapping[int, object],
    using: str,
) -> CanonicalRebuildResult:
    from django.db import models as django_models

    from apps.knowledge_graph.models import CanonicalEntity, CanonicalEntityLink

    projected_links = project_canonical_link_decisions(resolution)
    component_by_identity = {
        component.identity_key: component for component in resolution.components
    }
    if len(component_by_identity) != len(resolution.components):
        raise RuntimeError("canonical component identity is not unique")
    desired_identity_keys = tuple(
        sorted({row.target_identity_key for row in projected_links})
    )
    if any(key not in component_by_identity for key in desired_identity_keys):
        raise RuntimeError("projected canonical target escaped the resolution")
    projected_source_ids = {row.source_entity_id for row in projected_links}
    if not projected_source_ids.issubset(source_entities_by_id):
        raise RuntimeError("projected canonical source escaped the locked entity set")

    registry_rows: list[object] = list(
        _bounded_locked_rows(
            CanonicalEntity.objects.using(using)
            .select_for_update()
            .filter(resolver_version=resolution.resolver_version)
            .exclude(status=CanonicalEntity.Status.SUPERSEDED)
            .order_by("pk")
            .iterator(chunk_size=1_000),
            maximum=MAX_CANONICAL_ENTITIES,
            label="live canonical registry rows",
        )
    )
    for identity_batch in _batches(desired_identity_keys):
        _bounded_extend(
            registry_rows,
            CanonicalEntity.objects.using(using)
            .select_for_update()
            .filter(
                resolver_version=resolution.resolver_version,
                identity_key__in=identity_batch,
                status=CanonicalEntity.Status.SUPERSEDED,
            )
            .order_by("pk")
            .iterator(chunk_size=1_000),
            maximum=MAX_CANONICAL_ENTITIES * 2,
            label="desired historical canonical registry rows",
        )
    link_rows = _bounded_locked_rows(
        CanonicalEntityLink.objects.using(using)
        .select_for_update()
        .filter(resolver_version=resolution.resolver_version)
        .exclude(status=CanonicalEntityLink.Status.SUPERSEDED)
        .order_by("pk")
        .iterator(chunk_size=1_000),
        maximum=MAX_CANONICAL_MEMBERSHIPS,
        label="live canonical audit links",
    )
    registry_by_identity: dict[str, object] = {}
    for row in registry_rows:
        if row.identity_key in registry_by_identity:
            raise RuntimeError("canonical identity key is not unique")
        if row.status in {
            CanonicalEntity.Status.SUPPRESSED,
            CanonicalEntity.Status.REJECTED,
        }:
            raise RuntimeError("canonical registry contains a non-lifecycle status")
        registry_by_identity[row.identity_key] = row

    created_entities: list[object] = []
    canonical_by_identity: dict[str, object] = {}
    pending_entities: list[object] = []
    for identity_key in desired_identity_keys:
        component = component_by_identity[identity_key]
        canonical = registry_by_identity.get(identity_key)
        if canonical is not None:
            if (
                canonical.entity_type != component.entity_type
                or canonical.version_signature != component.version_signature
            ):
                raise RuntimeError(
                    "canonical registry identity conflicts with component type/version"
                )
        else:
            canonical = CanonicalEntity(
                identity_key=component.identity_key,
                resolver_version=resolution.resolver_version,
                label=component.label,
                normalized_label=component.normalized_label,
                entity_type=component.entity_type,
                version_signature=component.version_signature,
                status=CanonicalEntity.Status.ACTIVE,
                embedding=None,
                metadata={
                    "identity_method": component.method,
                    "registry_version": resolution.resolver_version,
                },
            )
            if canonical._raw_validation_errors():
                raise RuntimeError("generated canonical identity failed raw validation")
            canonical.clean()
            pending_entities.append(canonical)
        canonical_by_identity[identity_key] = canonical

    if pending_entities:
        django_models.QuerySet.bulk_create(
            CanonicalEntity.objects.using(using),
            pending_entities,
            batch_size=CANONICAL_QUERY_BATCH_SIZE,
        )
        if any(row.pk is None for row in pending_entities):
            raise RuntimeError("canonical bulk insert did not return primary keys")
        created_entities.extend(pending_entities)
        for canonical in pending_entities:
            registry_by_identity[canonical.identity_key] = canonical

    desired_canonical_ids = tuple(
        sorted({row.pk for row in canonical_by_identity.values()})
    )
    reactivate_ids = tuple(
        sorted(
            row.pk
            for row in canonical_by_identity.values()
            if row.status == CanonicalEntity.Status.SUPERSEDED
        )
    )
    if reactivate_ids:
        changed = CanonicalEntity._transition_registry_status_locked(
            reactivate_ids,
            target=CanonicalEntity.Status.ACTIVE,
            using=using,
        )
        if changed != len(reactivate_ids):
            raise RuntimeError("canonical registry reactivation changed concurrently")
        for row in canonical_by_identity.values():
            if row.pk in reactivate_ids:
                row.status = CanonicalEntity.Status.ACTIVE

    desired_links: dict[tuple[int, int], object] = {}
    for projected in projected_links:
        canonical = canonical_by_identity[projected.target_identity_key]
        source = source_entities_by_id[projected.source_entity_id]
        link = CanonicalEntityLink(
            collection_entity=source,
            canonical_entity=canonical,
            score=projected.score,
            method=projected.method,
            resolver_version=resolution.resolver_version,
            outcome=projected.outcome.value,
            status=projected.status,
            reason=projected.reason,
            metadata=json.loads(projected.metadata_json),
        )
        link.prepare_for_persistence()
        if link._raw_validation_errors():
            raise RuntimeError("generated canonical link failed raw validation")
        link.clean()
        key = (projected.source_entity_id, canonical.pk)
        if key in desired_links:
            raise RuntimeError("projected canonical link key is not unique")
        desired_links[key] = link

    current_by_key: dict[tuple[int, int], list[object]] = defaultdict(list)
    for row in link_rows:
        current_by_key[(row.collection_entity_id, row.canonical_entity_id)].append(row)
    keep_link_ids: set[int] = set()
    links_to_create: list[object] = []
    for key, desired in sorted(desired_links.items()):
        exact = tuple(
            row
            for row in current_by_key.get(key, ())
            if row.score == desired.score
            and row.method == desired.method
            and row.outcome == desired.outcome
            and row.status == desired.status
            and row.reason == desired.reason
            and row.metadata == desired.metadata
            and row.decision_checksum == desired.decision_checksum
        )
        if len(exact) > 1:
            raise RuntimeError("duplicate exact canonical audit links")
        if exact:
            keep_link_ids.add(exact[0].pk)
        else:
            links_to_create.append(desired)

    supersede_link_ids = tuple(
        sorted(row.pk for row in link_rows if row.pk not in keep_link_ids)
    )
    if supersede_link_ids:
        changed = CanonicalEntityLink._supersede_locked(supersede_link_ids, using=using)
        if changed != len(supersede_link_ids):
            raise RuntimeError("canonical link supersession changed concurrently")
    if links_to_create:
        django_models.QuerySet.bulk_create(
            CanonicalEntityLink.objects.using(using),
            links_to_create,
            batch_size=CANONICAL_QUERY_BATCH_SIZE,
        )

    supersede_entity_ids = tuple(
        sorted(
            row.pk
            for row in registry_rows
            if row.status == CanonicalEntity.Status.ACTIVE
            and row.pk not in desired_canonical_ids
        )
    )
    if supersede_entity_ids:
        changed = CanonicalEntity._transition_registry_status_locked(
            supersede_entity_ids,
            target=CanonicalEntity.Status.SUPERSEDED,
            using=using,
        )
        if changed != len(supersede_entity_ids):
            raise RuntimeError("canonical registry supersession changed concurrently")

    return CanonicalRebuildResult(
        resolver_version=resolution.resolver_version,
        resolution_checksum=resolution.checksum,
        active_artifact_ids=active_artifact_ids,
        canonical_entity_ids=desired_canonical_ids,
        active_link_count=sum(
            row.outcome is CanonicalOutcome.AUTOMATIC for row in projected_links
        ),
        created_entity_count=len(created_entities),
        created_link_count=len(links_to_create),
        superseded_entity_count=len(supersede_entity_ids),
        superseded_link_count=len(supersede_link_ids),
    )


def _rebuild_canonical_snapshot(
    snapshot: tuple[tuple[int, int], ...],
    *,
    using: str,
) -> CanonicalRebuildResult:
    from django.db import connections, transaction
    from django.db.models import F

    from apps.collections.models import Collection
    from apps.knowledge_graph.graph.assembly import _lock_collection_scope
    from apps.knowledge_graph.models import CollectionEntity, GraphArtifact

    collection_ids = tuple(sorted(collection_id for _, collection_id in snapshot))
    artifact_ids = tuple(sorted(artifact_id for artifact_id, _ in snapshot))
    with transaction.atomic(using=using):
        # Task 13 lock protocol: take the complete sorted C scope/row/artifact
        # prefix before the canonical advisory or any canonical/link row.
        with connections[using].cursor() as cursor:
            for collection_id in collection_ids:
                _lock_collection_scope(cursor, collection_id)
        locked_collections = tuple(
            Collection.objects.using(using)
            .select_for_update()
            .filter(pk__in=collection_ids)
            .order_by("pk")
        )
        if tuple(row.pk for row in locked_collections) != collection_ids:
            raise _CanonicalSnapshotChanged("active collection set changed")
        locked_artifacts = tuple(
            GraphArtifact.objects.using(using)
            .select_for_update()
            .filter(
                pk__in=artifact_ids,
                scope_type=GraphArtifact.ScopeType.COLLECTION,
                status=GraphArtifact.Status.ACTIVE,
            )
            .order_by("pk")
        )
        if tuple(row.pk for row in locked_artifacts) != artifact_ids:
            raise _CanonicalSnapshotChanged("active collection artifact set changed")
        expected_scope = {
            artifact_id: collection_id for artifact_id, collection_id in snapshot
        }
        if any(
            row.collection_scope_id != expected_scope[row.pk]
            or row.scope_id != str(expected_scope[row.pk])
            for row in locked_artifacts
        ):
            raise _CanonicalSnapshotChanged("active collection artifact scope changed")

        entity_rows = _bounded_locked_rows(
            CollectionEntity.objects.using(using)
            .select_for_update(of=("self",))
            .filter(
                artifact_id__in=artifact_ids,
                artifact__status=GraphArtifact.Status.ACTIVE,
                status=CollectionEntity.Status.ACTIVE,
                collection_id=F("artifact__collection_scope_id"),
            )
            .order_by("pk")
            .iterator(chunk_size=1_000),
            maximum=MAX_CANONICAL_ENTITIES,
            label="active collection entities",
        )
        inputs = _load_locked_canonical_inputs(
            entity_rows=entity_rows,
            active_artifact_ids=artifact_ids,
            using=using,
        )
        if _active_collection_artifact_snapshot(using=using) != snapshot:
            raise _CanonicalSnapshotChanged("canonical active snapshot expanded")
        resolution = resolve_canonical_entities(
            inputs,
            embedding_candidates=_derive_locked_embedding_candidates(
                entity_rows,
                artifact_embedding_signatures={
                    row.pk: row.embedding_model_signature for row in locked_artifacts
                },
            ),
        )

        _lock_canonical_registry(using=using)
        return _reconcile_locked_registry(
            resolution=resolution,
            active_artifact_ids=artifact_ids,
            source_entities_by_id={row.pk: row for row in entity_rows},
            using=using,
        )


def rebuild_canonical_registry(
    *,
    using: str = "default",
) -> CanonicalRebuildResult:
    """Atomically reconcile the full current registry with bounded drift retries."""

    last_error: _CanonicalSnapshotChanged | None = None
    for _attempt in range(_CANONICAL_REBUILD_ATTEMPTS):
        snapshot = _active_collection_artifact_snapshot(using=using)
        try:
            return _rebuild_canonical_snapshot(
                snapshot,
                using=using,
            )
        except _CanonicalSnapshotChanged as exc:
            last_error = exc
    raise RuntimeError("canonical active snapshot did not stabilize") from last_error


def schedule_canonical_rebuild(*, using: str = "default") -> None:
    """Register one robust post-commit full-registry reconciliation per transaction."""

    from django.db import transaction

    connection = transaction.get_connection(using)
    callback_marker = "_kg_canonical_rebuild_using"
    if any(
        getattr(entry[1], callback_marker, None) == using
        for entry in connection.run_on_commit
    ):
        return

    def reconcile() -> None:
        rebuild_canonical_registry(using=using)

    setattr(reconcile, callback_marker, using)
    transaction.on_commit(reconcile, using=using, robust=True)


def register_canonical_lifecycle_signals() -> None:
    """Rebuild after collection/canonical deletion without trusting stale caches."""

    from django.db.models.signals import post_delete

    from apps.knowledge_graph.models import (
        CanonicalEntity,
        CanonicalEntityLink,
        GraphArtifact,
    )

    def deleted(sender, instance, using="default", **_kwargs) -> None:
        if sender is GraphArtifact and (
            instance.scope_type != GraphArtifact.ScopeType.COLLECTION
        ):
            return
        schedule_canonical_rebuild(using=using)

    for sender in (GraphArtifact, CanonicalEntityLink, CanonicalEntity):
        post_delete.connect(
            deleted,
            sender=sender,
            weak=False,
            dispatch_uid=(
                f"apps_knowledge_graph.canonical.post_delete.{sender._meta.label_lower}"
            ),
        )


__all__ = [
    "CANONICAL_RESOLVER_VERSION",
    "CanonicalAcronymExpansion",
    "CanonicalAliasEvidence",
    "CanonicalComponent",
    "CanonicalDecision",
    "CanonicalEmbeddingCandidate",
    "CanonicalEntityInput",
    "CanonicalLookupResult",
    "CanonicalMembershipRow",
    "CanonicalOutcome",
    "CanonicalProjectedLinkDecision",
    "CanonicalProvenanceRow",
    "CanonicalRebuildResult",
    "CanonicalResolutionResult",
    "CanonicalSourceMembership",
    "PermissionBearingEntity",
    "authorized_canonical_lookup",
    "build_canonical_inputs_from_provenance",
    "project_authorized_canonical_lookup",
    "project_canonical_link_decisions",
    "register_canonical_lifecycle_signals",
    "rebuild_canonical_registry",
    "resolve_canonical_entities",
    "schedule_canonical_rebuild",
]
