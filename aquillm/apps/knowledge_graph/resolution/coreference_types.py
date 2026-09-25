"""Immutable mention, decision, and cluster values for coreference."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .coreference_validation import (
    _canonical_uuid,
    _confidence,
    _mention_key,
    _require_string,
    _validated_coordinate_basis,
    _validated_db_integer,
    _validated_entity_type,
    _validated_identifier,
    _validated_source_key,
    _validated_source_text,
    _validated_span,
)
from .normalization import parse_stable_identifier

_HASH = re.compile(r"[0-9a-f]{64}")
_VERSION_SIGNATURE = re.compile(r"[a-z0-9][a-z0-9.+:/_-]*")
@dataclass(frozen=True, slots=True)
class DocumentMention:
    """Provider-neutral mention input with optional bounded source context."""

    mention_id: object
    raw_text: str
    entity_type: str
    start: int
    end: int
    source_text: str = ""
    source_offset: int = 0
    identifier: str = ""
    confidence: float = 1.0
    document_id: object = ""
    source_key: str = ""
    chunk_id: object = ""
    position_basis: str = "document_global"
    content_object_id: object | None = None

    def __post_init__(self) -> None:
        _mention_key(self.mention_id)
        _require_string(self.raw_text, "raw_text")
        _validated_entity_type(self.entity_type)
        _validated_span(self.start, self.end)
        _validated_source_text(self.source_text)
        _validated_db_integer(self.source_offset, "source_offset", minimum=0)
        _validated_identifier(self.identifier)
        if type(self.source_key) is not str:
            raise ValueError("source key must be a nonempty string")
        if self.source_key != "":
            _validated_source_key(self.source_key)
        _canonical_uuid(self.document_id, "document_id")
        _validated_db_integer(self.chunk_id, "chunk_id", minimum=1)
        _validated_coordinate_basis(self.position_basis, self.content_object_id)
        _confidence(self.confidence)


@dataclass(frozen=True, slots=True)
class PairDecision:
    """One direct auditable decision for an unordered mention pair."""

    left_mention_id: str
    right_mention_id: str
    accepted: bool
    method: str
    confidence: float
    explanation: str

    def __post_init__(self) -> None:
        _require_string(self.left_mention_id, "left_mention_id")
        _require_string(self.right_mention_id, "right_mention_id")
        if self.left_mention_id == self.right_mention_id:
            raise ValueError("a pair decision requires distinct mentions")
        if type(self.accepted) is not bool:
            raise ValueError("accepted must be a boolean")
        _require_string(self.method, "method")
        _confidence(self.confidence)
        _require_string(self.explanation, "explanation")


@dataclass(frozen=True, slots=True)
class ClusterMembership:
    """One mention's deterministic parent edge inside a resolved cluster."""

    mention_id: str
    method: str
    reason: str
    parent_mention_id: str | None

    def __post_init__(self) -> None:
        _require_string(self.mention_id, "mention_id")
        _require_string(self.method, "method")
        _require_string(self.reason, "reason")
        if self.parent_mention_id is not None:
            _require_string(self.parent_mention_id, "parent_mention_id")
            if self.parent_mention_id == self.mention_id:
                raise ValueError("membership cannot parent itself")


@dataclass(frozen=True, slots=True)
class ResolvedCluster:
    """One immutable document entity candidate and its mention membership."""

    cluster_key: str
    mention_ids: tuple[str, ...]
    memberships: tuple[ClusterMembership, ...]
    label: str
    normalized_label: str
    version_signature: str
    entity_type: str
    identifier: str
    method: str
    confidence: float

    def __post_init__(self) -> None:
        if type(self.cluster_key) is not str or not _HASH.fullmatch(self.cluster_key):
            raise ValueError("cluster_key must be a lowercase SHA-256 digest")
        if type(self.mention_ids) is not tuple or not self.mention_ids:
            raise ValueError("mention_ids must be a nonempty tuple")
        if not all(type(mention_id) is str for mention_id in self.mention_ids):
            raise ValueError("mention_ids must contain exact strings")
        if len(set(self.mention_ids)) != len(self.mention_ids):
            raise ValueError("cluster mention IDs must be unique")
        if type(self.memberships) is not tuple or not all(
            isinstance(membership, ClusterMembership) for membership in self.memberships
        ):
            raise ValueError("memberships must contain ClusterMembership values")
        membership_ids = tuple(item.mention_id for item in self.memberships)
        if len(set(membership_ids)) != len(membership_ids) or set(
            membership_ids
        ) != set(self.mention_ids):
            raise ValueError(
                "memberships must describe every cluster mention exactly once"
            )
        for field_name in ("label", "normalized_label", "entity_type", "method"):
            _require_string(getattr(self, field_name), field_name)
        if (
            type(self.version_signature) is not str
            or len(self.version_signature) > 128
            or (
                self.version_signature
                and not _VERSION_SIGNATURE.fullmatch(self.version_signature)
            )
        ):
            raise ValueError("version_signature must be blank or canonical lower ASCII")
        if type(self.identifier) is not str:
            raise ValueError("identifier must be a string")
        if len(self.identifier) > 255:
            raise ValueError("identifier exceeds the persistence limit")
        if self.identifier:
            parsed_identifier = parse_stable_identifier(self.identifier)
            if (
                parsed_identifier is None
                or parsed_identifier.canonical != self.identifier
            ):
                raise ValueError(
                    "identifier must be an exact canonical stable identifier"
                )
        _confidence(self.confidence)
        membership_by_id = {item.mention_id: item for item in self.memberships}
        roots = [item for item in self.memberships if item.parent_mention_id is None]
        expected_root_method = "singleton" if len(self.mention_ids) == 1 else "root"
        if len(roots) != 1 or roots[0].method != expected_root_method:
            raise ValueError("cluster memberships require exactly one explicit root")
        root_id = roots[0].mention_id
        for membership in self.memberships:
            if membership.mention_id == root_id:
                continue
            if (
                membership.parent_mention_id not in membership_by_id
                or membership.method in {"root", "singleton"}
            ):
                raise ValueError("non-root membership requires a valid parent edge")
            seen: set[str] = set()
            cursor = membership
            while cursor.parent_mention_id is not None:
                if cursor.mention_id in seen:
                    raise ValueError("cluster membership parents must be acyclic")
                seen.add(cursor.mention_id)
                cursor = membership_by_id[cursor.parent_mention_id]
            if cursor.mention_id != root_id:
                raise ValueError("every membership parent path must reach the root")


@dataclass(frozen=True, slots=True)
class _MentionView:
    mention_id: str
    raw_text: str
    display_label: str
    normalized_label: str
    base_key: str
    version_signature: str | None
    raw_entity_type: str
    entity_type: str
    start: int
    end: int
    source_text: str
    source_offset: int
    document_id: str
    source_key: str
    coordinate_scope: str
    chunk_id: str
    position_basis: str
    content_object_id: str
    member_key: str
    identifier: str
    confidence: float
    is_acronym: bool
    acronym_shape_key: str
    is_pronoun: bool

    @property
    def sort_key(self) -> tuple[object, ...]:
        return (
            self.member_key,
            self.start,
            self.end,
            self.entity_type,
            self.normalized_label,
            self.mention_id,
        )


def _is_count(value: object) -> bool:
    return type(value) is int and value >= 0


def _is_hash(value: object) -> bool:
    return type(value) is str and _HASH.fullmatch(value) is not None
