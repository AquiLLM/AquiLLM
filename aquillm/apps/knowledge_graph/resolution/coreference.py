"""Pure, bounded, conservative within-document mention clustering."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from hashlib import sha256
from itertools import combinations, islice
from math import isfinite
from uuid import UUID

from . import DOCUMENT_RESOLVER_VERSION
from .normalization import normalize_entity_label, parse_stable_identifier

MAX_DOCUMENT_MENTIONS = 512
_MAX_SOURCE_TEXT_CHARACTERS = 1_000_000
_MAX_UNIQUE_SOURCE_CONTEXT_CHARACTERS = 2_000_000
_MAX_IDENTIFIER_CHARACTERS = 2_048
_MAX_SOURCE_KEY_CHARACTERS = 512
_MAX_MENTION_ID_CHARACTERS = 128
_MAX_ENTITY_TYPE_CHARACTERS = 128
_HASH = re.compile(r"[0-9a-f]{64}")
_VERSION_SIGNATURE = re.compile(r"[a-z0-9][a-z0-9.+:/_-]*")
_MISSING = object()
_ACRONYM = re.compile(r"[A-Z][A-Z0-9-]{1,11}")
_WORD = re.compile(r"[A-Za-z0-9]+")
_INITIALISM_STOPWORDS = frozenset(
    ("a", "an", "and", "for", "from", "in", "of", "on", "the", "to", "with")
)
_PRONOUN_LABELS = frozenset(
    (
        "he",
        "her",
        "hers",
        "him",
        "his",
        "it",
        "its",
        "she",
        "their",
        "theirs",
        "them",
        "they",
        "this approach",
        "this dataset",
        "this method",
        "this model",
        "this system",
        "we",
    )
)
_METHOD_PRECEDENCE = {
    "stable_identifier": 0,
    "defined_acronym": 1,
    "ontology_alias": 2,
    "normalized_name": 3,
    "singleton": 4,
}
_HARD_CANNOT_LINK_METHODS = frozenset(
    (
        "ambiguous_acronym",
        "component_conflict",
        "conflicting_stable_identifiers",
        "incompatible_entity_types",
        "lowercase_acronym",
        "pre_definition_acronym",
        "pronoun_only",
        "source_mismatch",
        "undefined_acronym",
        "version_mismatch",
    )
)


def _require_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a nonempty string")
    return value


def _contains_unsafe_control(
    value: str,
    *,
    allow_text_whitespace: bool,
    allow_format_controls: bool = False,
) -> bool:
    allowed = {"\t", "\n", "\r"} if allow_text_whitespace else set()
    for character in value:
        category = unicodedata.category(character)
        if character not in allowed and category in {"Cc", "Cs"}:
            return True
        if not allow_format_controls and category == "Cf":
            return True
    return False


def _mention_key(value: object) -> str:
    if isinstance(value, bool) or value is None:
        raise ValueError("mention_id must be a stable nonempty scalar")
    try:
        key = str(value).strip()
    except (OverflowError, ValueError) as exc:
        raise ValueError("mention_id must be a stable nonempty scalar") from exc
    if not key:
        raise ValueError("mention_id must be a stable nonempty scalar")
    if len(key) > _MAX_MENTION_ID_CHARACTERS:
        raise ValueError(
            f"mention_id exceeds the {_MAX_MENTION_ID_CHARACTERS}-character limit"
        )
    if _contains_unsafe_control(key, allow_text_whitespace=False):
        raise ValueError("mention_id contains an unsafe control character")
    return key


def _confidence(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("mention confidence must be a finite confidence in [0, 1]")
    try:
        converted = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(
            "mention confidence must be a finite confidence in [0, 1]"
        ) from exc
    if not isfinite(converted) or not 0 <= converted <= 1:
        raise ValueError("mention confidence must be a finite confidence in [0, 1]")
    return converted


def _validated_source_text(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("source text must be a string")
    if len(value) > _MAX_SOURCE_TEXT_CHARACTERS:
        raise ValueError(
            f"source text exceeds the {_MAX_SOURCE_TEXT_CHARACTERS}-character limit"
        )
    if _contains_unsafe_control(
        value,
        allow_text_whitespace=True,
        allow_format_controls=True,
    ):
        raise ValueError("source text contains an unsafe control character")
    return value


def _validated_identifier(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("identifier must be a string")
    if len(value) > _MAX_IDENTIFIER_CHARACTERS:
        raise ValueError(
            f"identifier exceeds the {_MAX_IDENTIFIER_CHARACTERS}-character limit"
        )
    if _contains_unsafe_control(value, allow_text_whitespace=False):
        raise ValueError("identifier contains an unsafe control character")
    return value


def _validated_source_key(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("source key must be a nonempty string")
    if len(value) > _MAX_SOURCE_KEY_CHARACTERS:
        raise ValueError(
            f"source key exceeds the {_MAX_SOURCE_KEY_CHARACTERS}-character limit"
        )
    if _contains_unsafe_control(value, allow_text_whitespace=False):
        raise ValueError("source key contains an unsafe control character")
    return value


def _validated_entity_type(value: object) -> str:
    entity_type = _require_string(value, "entity_type")
    if len(entity_type) > _MAX_ENTITY_TYPE_CHARACTERS:
        raise ValueError(
            f"entity_type exceeds the {_MAX_ENTITY_TYPE_CHARACTERS}-character limit"
        )
    if _contains_unsafe_control(entity_type, allow_text_whitespace=False):
        raise ValueError("entity_type contains an unsafe control character")
    return entity_type


def _canonical_uuid(value: object, field_name: str, *, optional: bool = False) -> str:
    if optional and (value is None or (isinstance(value, str) and value == "")):
        return ""
    if not isinstance(value, (str, UUID)):
        raise ValueError(f"{field_name} must be a UUID")
    if isinstance(value, str) and len(value) > 64:
        raise ValueError(f"{field_name} exceeds the 64-character UUID limit")
    try:
        parsed = value if isinstance(value, UUID) else UUID(str(value))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a UUID") from exc
    return str(parsed)


def _positive_chunk_id(value: object) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError("chunk_id must be a positive integer")
    return value


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
        if type(self.start) is not int or self.start < 0:
            raise ValueError("start must be a nonnegative integer")
        if type(self.end) is not int or self.end <= self.start:
            raise ValueError("end must be greater than start")
        _validated_source_text(self.source_text)
        if type(self.source_offset) is not int or self.source_offset < 0:
            raise ValueError("source_offset must be a nonnegative integer")
        _validated_identifier(self.identifier)
        if self.source_key != "":
            _validated_source_key(self.source_key)
        _canonical_uuid(self.document_id, "document_id")
        _positive_chunk_id(self.chunk_id)
        _canonical_uuid(self.content_object_id, "content_object_id", optional=True)
        if self.position_basis not in {"document_global", "chunk_content"}:
            raise ValueError("position_basis must be document_global or chunk_content")
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
        if not _HASH.fullmatch(self.cluster_key):
            raise ValueError("cluster_key must be a lowercase SHA-256 digest")
        if not isinstance(self.mention_ids, tuple) or not self.mention_ids:
            raise ValueError("mention_ids must be a nonempty tuple")
        if len(set(self.mention_ids)) != len(self.mention_ids):
            raise ValueError("cluster mention IDs must be unique")
        if not isinstance(self.memberships, tuple) or not all(
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
            not isinstance(self.version_signature, str)
            or len(self.version_signature) > 128
            or (
                self.version_signature
                and not _VERSION_SIGNATURE.fullmatch(self.version_signature)
            )
        ):
            raise ValueError("version_signature must be blank or canonical lower ASCII")
        if not isinstance(self.identifier, str):
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
class ResolutionResult:
    """Complete immutable partition and direct decision audit."""

    resolver_version: str
    ontology_checksum: str
    input_fingerprint: str
    mention_ids: tuple[str, ...]
    clusters: tuple[ResolvedCluster, ...]
    decisions: tuple[PairDecision, ...]
    checksum: str

    def __post_init__(self) -> None:
        _require_string(self.resolver_version, "resolver_version")
        if not _HASH.fullmatch(self.ontology_checksum):
            raise ValueError("ontology_checksum must be a lowercase SHA-256 digest")
        if not _HASH.fullmatch(self.input_fingerprint):
            raise ValueError("input_fingerprint must be a lowercase SHA-256 digest")
        if not isinstance(self.mention_ids, tuple):
            raise ValueError("mention_ids must be a tuple")
        if len(set(self.mention_ids)) != len(self.mention_ids):
            raise ValueError("result mention IDs must be unique")
        if not isinstance(self.clusters, tuple) or not all(
            isinstance(cluster, ResolvedCluster) for cluster in self.clusters
        ):
            raise ValueError("clusters must contain ResolvedCluster values")
        cluster_keys = tuple(cluster.cluster_key for cluster in self.clusters)
        if len(set(cluster_keys)) != len(cluster_keys):
            raise ValueError("cluster keys must be unique")
        if not isinstance(self.decisions, tuple) or not all(
            isinstance(decision, PairDecision) for decision in self.decisions
        ):
            raise ValueError("decisions must contain PairDecision values")
        memberships = tuple(
            mention_id
            for cluster in self.clusters
            for mention_id in cluster.mention_ids
        )
        if sorted(memberships) != sorted(self.mention_ids):
            raise ValueError("clusters must partition all result mentions exactly once")
        expected_pairs = {frozenset(pair) for pair in combinations(self.mention_ids, 2)}
        actual_pairs = {
            frozenset((decision.left_mention_id, decision.right_mention_id))
            for decision in self.decisions
        }
        if len(actual_pairs) != len(self.decisions) or actual_pairs != expected_pairs:
            raise ValueError("decisions must audit every mention pair exactly once")
        if not _HASH.fullmatch(self.checksum):
            raise ValueError("checksum must be a lowercase SHA-256 digest")


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


@dataclass(frozen=True, slots=True)
class _AcronymDefinition:
    expansion: str
    source_key: str
    full_mention_id: str
    acronym_mention_id: str
    definition_start: int


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self._parents = list(range(size))

    def find(self, item: int) -> int:
        parent = self._parents[item]
        while parent != self._parents[parent]:
            self._parents[parent] = self._parents[self._parents[parent]]
            parent = self._parents[parent]
        self._parents[item] = parent
        return parent

    def union(self, left: int, right: int) -> int:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return left_root
        smaller, larger = sorted((left_root, right_root))
        self._parents[larger] = smaller
        return smaller


def _value(source: object, name: str, default: object = None) -> object:
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


def _ontology_type_index(ontology: object) -> tuple[dict[str, str], str]:
    entity_types = _value(ontology, "entity_types")
    if not isinstance(entity_types, Mapping) or not entity_types:
        raise ValueError("ontology.entity_types must be a nonempty mapping")
    index: dict[str, str] = {}
    checksum_records: list[dict[str, object]] = []
    for map_name, definition in sorted(
        entity_types.items(), key=lambda item: str(item[0])
    ):
        validated_map_name = _validated_ontology_type_text(
            map_name, "ontology type map key"
        )
        name = _value(definition, "name", map_name)
        canonical = _validated_ontology_type_text(name, "ontology type name")
        aliases = _value(definition, "aliases", ())
        if not isinstance(aliases, (tuple, list)):
            raise ValueError("ontology type aliases must be a tuple or list")
        validated_aliases = tuple(
            _validated_ontology_type_text(alias, "ontology type alias")
            for alias in aliases
        )
        for alias in (validated_map_name, canonical, *validated_aliases):
            previous = index.get(alias)
            if previous is not None and previous != canonical:
                raise ValueError(f"ambiguous ontology type alias: {alias}")
            index[alias] = canonical
        checksum_records.append(
            {
                "map_name": validated_map_name,
                "name": canonical,
                "aliases": sorted(validated_aliases),
            }
        )
    persisted_checksum = _value(ontology, "checksum", "")
    if persisted_checksum:
        if not isinstance(persisted_checksum, str) or not _HASH.fullmatch(
            persisted_checksum
        ):
            raise ValueError("ontology checksum must be a lowercase SHA-256 digest")
        checksum = persisted_checksum
    else:
        encoded = json.dumps(
            checksum_records,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        checksum = sha256(encoded).hexdigest()
    return index, checksum


def _validated_ontology_type_text(value: object, field_name: str) -> str:
    raw = _require_string(value, field_name)
    if len(raw) > _MAX_ENTITY_TYPE_CHARACTERS:
        raise ValueError(
            f"{field_name} exceeds the {_MAX_ENTITY_TYPE_CHARACTERS}-character limit"
        )
    if _contains_unsafe_control(raw, allow_text_whitespace=False):
        raise ValueError(f"{field_name} contains an unsafe control character")
    normalized = " ".join(unicodedata.normalize("NFKC", raw).casefold().split())
    if not normalized:
        raise ValueError(f"{field_name} must be a nonempty string")
    if len(normalized) > _MAX_ENTITY_TYPE_CHARACTERS:
        raise ValueError(
            f"{field_name} exceeds the {_MAX_ENTITY_TYPE_CHARACTERS}-character limit"
        )
    return normalized


def _metadata_identifier(mention: object) -> object:
    direct = _value(mention, "identifier", "")
    if direct != "":
        return _validated_identifier(direct)
    metadata = _value(mention, "metadata", {})
    if not isinstance(metadata, Mapping):
        return ""
    candidate = metadata.get("stable_identifier") or metadata.get("identifier") or ""
    return _validated_identifier(candidate)


def _source_context(mention: object) -> tuple[str, int]:
    explicit_text = _value(mention, "source_text", _MISSING)
    explicit_offset = _value(mention, "source_offset", 0)
    if explicit_text is not _MISSING:
        if type(explicit_offset) is not int or explicit_offset < 0:
            raise ValueError("mention source context is invalid")
        return _validated_source_text(explicit_text), explicit_offset
    chunk = _value(mention, "chunk")
    if chunk is None:
        return "", 0
    content = _value(chunk, "content", "")
    basis = _value(mention, "position_basis", "document_global")
    offset = _value(chunk, "start_position", 0) if basis == "document_global" else 0
    if type(offset) is not int or offset < 0:
        raise ValueError("mention chunk context is invalid")
    return _validated_source_text(content), offset


def _source_identity(mention: object) -> tuple[str, str, str, str, str]:
    document_id = _canonical_uuid(_value(mention, "document_id"), "document_id")
    chunk_id = _positive_chunk_id(_value(mention, "chunk_id"))
    position_basis = _value(mention, "position_basis", "document_global")
    content_object_id = _canonical_uuid(
        _value(mention, "content_object_id"),
        "content_object_id",
        optional=True,
    )
    explicit_source_key = _value(mention, "source_key", _MISSING)
    if position_basis not in {"document_global", "chunk_content"}:
        raise ValueError("mention position_basis is invalid")
    if explicit_source_key is not _MISSING and explicit_source_key != "":
        source_key = _validated_source_key(explicit_source_key)
    elif position_basis == "document_global":
        source_key = f"document:{document_id}"
    elif content_object_id:
        source_key = f"content:{content_object_id}"
    else:
        source_key = f"chunk:{chunk_id}"
    return (
        document_id,
        _validated_source_key(source_key),
        str(chunk_id),
        position_basis,
        content_object_id,
    )


def _member_key(
    *,
    document_id: str,
    source_key: str,
    position_basis: str,
    content_object_id: str,
    start: int,
    end: int,
    entity_type: str,
    normalized_label: str,
) -> str:
    payload = {
        "document_id": document_id,
        "source_key": source_key,
        "position_basis": position_basis,
        "content_object_id": content_object_id,
        "start": start,
        "end": end,
        "entity_type": entity_type,
        "normalized_label": normalized_label,
    }
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _adapt_mention(mention: object, type_index: Mapping[str, str]) -> _MentionView:
    mention_id = _value(mention, "mention_id")
    if mention_id is None:
        mention_id = _value(mention, "pk", _value(mention, "id"))
    raw_text = _value(mention, "raw_text", _value(mention, "text"))
    raw_entity_type = _value(mention, "entity_type")
    start = _value(mention, "start")
    end = _value(mention, "end")
    confidence = _value(
        mention,
        "confidence",
        _value(mention, "extraction_confidence", 1.0),
    )
    normalized = normalize_entity_label(raw_text)
    raw_type = unicodedata.normalize("NFKC", _validated_entity_type(raw_entity_type))
    raw_type = " ".join(raw_type.casefold().split())
    canonical_type = type_index.get(raw_type)
    if canonical_type is None:
        raise ValueError(f"unknown ontology entity type: {raw_type}")
    if type(start) is not int or start < 0 or type(end) is not int or end <= start:
        raise ValueError("mention span must be nonnegative and nonempty")
    source_text, source_offset = _source_context(mention)
    (
        document_id,
        source_key,
        chunk_id,
        position_basis,
        content_object_id,
    ) = _source_identity(mention)
    explicit_identifier = _metadata_identifier(mention)
    if explicit_identifier:
        parsed_identifier = parse_stable_identifier(explicit_identifier)
        if parsed_identifier is None:
            raise ValueError("explicit identifier must be a valid stable identifier")
    else:
        parsed_identifier = parse_stable_identifier(raw_text)
    display = normalized.display_label
    return _MentionView(
        mention_id=_mention_key(mention_id),
        raw_text=raw_text,
        display_label=display,
        normalized_label=normalized.key,
        base_key=normalized.base_key,
        version_signature=normalized.version_signature,
        raw_entity_type=raw_type,
        entity_type=canonical_type,
        start=start,
        end=end,
        source_text=source_text,
        source_offset=source_offset,
        document_id=document_id,
        source_key=source_key,
        chunk_id=chunk_id,
        position_basis=position_basis,
        content_object_id=content_object_id,
        member_key=_member_key(
            document_id=document_id,
            source_key=source_key,
            position_basis=position_basis,
            content_object_id=content_object_id,
            start=start,
            end=end,
            entity_type=canonical_type,
            normalized_label=normalized.key,
        ),
        identifier=parsed_identifier.canonical if parsed_identifier else "",
        confidence=_confidence(confidence),
        is_acronym=_is_acronym(display),
        acronym_shape_key=_acronym_shape_key(display),
        is_pronoun=normalized.key in _PRONOUN_LABELS,
    )


def resolution_input_fingerprint(mentions: Iterable[object]) -> str:
    """Bind a result to exact mention fields and local source context."""

    bounded = tuple(islice(iter(mentions), MAX_DOCUMENT_MENTIONS + 1))
    if len(bounded) > MAX_DOCUMENT_MENTIONS:
        raise ValueError(f"document mention cap exceeded ({MAX_DOCUMENT_MENTIONS})")
    records: list[dict[str, object]] = []
    source_contexts: dict[tuple[str, int, str], dict[str, object]] = {}
    aggregate_source_characters = 0
    document_id_seen: str | None = None
    for mention in bounded:
        mention_id = _value(mention, "mention_id")
        if mention_id is None:
            mention_id = _value(mention, "pk", _value(mention, "id"))
        raw_text = _value(mention, "raw_text", _value(mention, "text"))
        entity_type = _value(mention, "entity_type")
        start = _value(mention, "start")
        end = _value(mention, "end")
        confidence = _value(
            mention,
            "confidence",
            _value(mention, "extraction_confidence", 1.0),
        )
        validated_raw_text = _require_string(raw_text, "raw_text")
        normalize_entity_label(validated_raw_text)
        validated_entity_type = _validated_entity_type(entity_type)
        document_id, source_key, chunk_id, basis, content_id = _source_identity(mention)
        if document_id_seen is None:
            document_id_seen = document_id
        elif document_id != document_id_seen:
            raise ValueError("mentions must belong to a single document")
        source_text, source_offset = _source_context(mention)
        context_key = (source_key, source_offset, source_text)
        context_record = source_contexts.get(context_key)
        if context_record is None:
            aggregate_source_characters += len(source_text)
            if aggregate_source_characters > _MAX_UNIQUE_SOURCE_CONTEXT_CHARACTERS:
                raise ValueError(
                    "aggregate unique source context exceeds the "
                    f"{_MAX_UNIQUE_SOURCE_CONTEXT_CHARACTERS}-character limit"
                )
            context_digest = _source_context_digest(
                source_key=source_key,
                source_offset=source_offset,
                source_text=source_text,
            )
            context_record = {
                "digest": context_digest,
                "source_key": source_key,
                "source_offset": source_offset,
                "character_count": len(source_text),
            }
            source_contexts[context_key] = context_record
        records.append(
            {
                "mention_id": _mention_key(mention_id),
                "document_id": document_id,
                "source_key": source_key,
                "chunk_id": chunk_id,
                "position_basis": basis,
                "content_object_id": content_id,
                "start": start,
                "end": end,
                "raw_text": validated_raw_text,
                "entity_type": validated_entity_type,
                "identifier": str(_metadata_identifier(mention) or ""),
                "confidence": _confidence(confidence),
                "source_context_digest": context_record["digest"],
            }
        )
    records.sort(
        key=lambda item: json.dumps(
            item,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    )
    mention_ids = [record["mention_id"] for record in records]
    if len(set(mention_ids)) != len(mention_ids):
        raise ValueError("mention IDs must be unique within a document")
    payload = {
        "source_contexts": sorted(
            source_contexts.values(),
            key=lambda item: (
                item["source_key"],
                item["source_offset"],
                item["digest"],
            ),
        ),
        "mentions": records,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _source_context_digest(
    *, source_key: str, source_offset: int, source_text: str
) -> str:
    digest = sha256()
    for value in (source_key, str(source_offset), source_text):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _is_acronym(value: str) -> bool:
    compact = unicodedata.normalize("NFKC", value).strip()
    return bool(
        _ACRONYM.fullmatch(compact)
        and sum(character.isalpha() for character in compact) >= 2
        and compact == compact.upper()
    )


def _acronym_shape_key(value: str) -> str:
    compact = unicodedata.normalize("NFKC", value).strip()
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9-]{1,11}", compact):
        return ""
    if sum(character.isalpha() for character in compact) < 2:
        return ""
    return _acronym_key(compact)


def _acronym_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    return "".join(character for character in normalized.upper() if character.isalnum())


def _initialism(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    words = [
        word
        for word in _WORD.findall(value)
        if word.casefold() not in _INITIALISM_STOPWORDS
    ]
    return "".join(word[0].upper() for word in words)


def _is_parenthetical_definition(full: _MentionView, acronym: _MentionView) -> bool:
    if (
        not full.source_text
        or full.source_text != acronym.source_text
        or full.source_key != acronym.source_key
        or full.source_offset != acronym.source_offset
        or full.end >= acronym.start
        or _initialism(full.display_label) != _acronym_key(acronym.display_label)
    ):
        return False
    full_end = full.end - full.source_offset
    acronym_start = acronym.start - acronym.source_offset
    acronym_end = acronym.end - acronym.source_offset
    source = full.source_text
    if not (0 <= full_end <= acronym_start < acronym_end <= len(source)):
        return False
    between = source[full_end:acronym_start]
    following = source[acronym_end:]
    return bool(re.fullmatch(r"\s*\(\s*", between) and re.match(r"\s*\)", following))


def _acronym_expansions(
    mentions: tuple[_MentionView, ...],
) -> dict[tuple[str, str], tuple[_AcronymDefinition, ...]]:
    definitions: dict[tuple[str, str], list[_AcronymDefinition]] = defaultdict(list)
    full_mentions = [mention for mention in mentions if not mention.is_acronym]
    acronym_mentions = [mention for mention in mentions if mention.is_acronym]
    for full in full_mentions:
        for acronym in acronym_mentions:
            if full.entity_type == acronym.entity_type and _is_parenthetical_definition(
                full, acronym
            ):
                definitions[
                    (acronym.entity_type, _acronym_key(acronym.display_label))
                ].append(
                    _AcronymDefinition(
                        expansion=full.normalized_label,
                        source_key=full.source_key,
                        full_mention_id=full.mention_id,
                        acronym_mention_id=acronym.mention_id,
                        definition_start=acronym.start,
                    )
                )
    return {
        key: tuple(
            sorted(
                values,
                key=lambda item: (
                    item.source_key,
                    item.definition_start,
                    item.full_mention_id,
                    item.acronym_mention_id,
                ),
            )
        )
        for key, values in definitions.items()
    }


def _name_identifier_conflicts(
    mentions: tuple[_MentionView, ...],
) -> frozenset[tuple[str, str]]:
    identifiers: dict[tuple[str, str], set[str]] = defaultdict(set)
    for mention in mentions:
        if mention.identifier:
            identifiers[(mention.entity_type, mention.normalized_label)].add(
                mention.identifier
            )
    return frozenset(key for key, values in identifiers.items() if len(values) > 1)


def _rejected(left: _MentionView, right: _MentionView, method: str, reason: str):
    return PairDecision(
        left.mention_id,
        right.mention_id,
        False,
        method,
        0.0,
        f"Rejected: {reason}.",
    )


def _accepted(left: _MentionView, right: _MentionView, method: str, reason: str):
    return PairDecision(
        left.mention_id,
        right.mention_id,
        True,
        method,
        min(left.confidence, right.confidence),
        f"Accepted: {reason}.",
    )


def _acronym_decision(
    left: _MentionView,
    right: _MentionView,
    expansions: Mapping[tuple[str, str], tuple[_AcronymDefinition, ...]],
) -> PairDecision | None:
    left_definitions = expansions.get((left.entity_type, left.acronym_shape_key), ())
    right_definitions = expansions.get((right.entity_type, right.acronym_shape_key), ())
    left_is_candidate = left.is_acronym or bool(left_definitions)
    right_is_candidate = right.is_acronym or bool(right_definitions)
    if not (left_is_candidate or right_is_candidate):
        return None
    if (left_definitions and not left.is_acronym) or (
        right_definitions and not right.is_acronym
    ):
        return _rejected(
            left,
            right,
            "lowercase_acronym",
            "lowercase acronym occurrences are not resolved",
        )
    if left_is_candidate and right_is_candidate:
        left_key = left.acronym_shape_key
        right_key = right.acronym_shape_key
        if left_key != right_key:
            return None
        definitions = expansions.get((left.entity_type, left_key), ())
        candidate_expansions = {item.expansion for item in definitions}
        if len(candidate_expansions) > 1:
            return _rejected(left, right, "ambiguous_acronym", "ambiguous acronym")
        if not definitions:
            return _rejected(left, right, "undefined_acronym", "undefined acronym")
        if left.source_key != right.source_key:
            return _rejected(
                left,
                right,
                "source_mismatch",
                "acronym definition belongs to another source coordinate space",
            )

        def is_resolved(mention: _MentionView) -> bool:
            return any(
                definition.source_key == mention.source_key
                and mention.start >= definition.definition_start
                for definition in definitions
            )

        if is_resolved(left) and is_resolved(right):
            return _accepted(
                left, right, "normalized_name", "same uniquely defined acronym"
            )
        return _rejected(
            left,
            right,
            "pre_definition_acronym",
            "acronym occurrence precedes its definition",
        )
    acronym, full = (left, right) if left_is_candidate else (right, left)
    acronym_key = acronym.acronym_shape_key
    if _initialism(full.display_label) != acronym_key:
        return None
    definitions = expansions.get((acronym.entity_type, acronym_key), ())
    candidate_expansions = {item.expansion for item in definitions}
    if len(candidate_expansions) > 1:
        return _rejected(left, right, "ambiguous_acronym", "ambiguous acronym")
    if not definitions:
        return _rejected(left, right, "undefined_acronym", "undefined acronym")
    relevant = tuple(
        definition
        for definition in definitions
        if definition.expansion == full.normalized_label
    )
    if not relevant:
        return None
    same_source = tuple(
        definition
        for definition in relevant
        if definition.source_key == acronym.source_key
    )
    if not same_source:
        return _rejected(
            left,
            right,
            "source_mismatch",
            "acronym definition belongs to another source coordinate space",
        )
    if any(acronym.start >= item.definition_start for item in same_source):
        return _accepted(
            left,
            right,
            "defined_acronym",
            "document explicitly defines this acronym expansion",
        )
    return _rejected(
        left,
        right,
        "pre_definition_acronym",
        "acronym occurrence precedes its definition",
    )


def _decide_pair(
    left: _MentionView,
    right: _MentionView,
    *,
    expansions: Mapping[tuple[str, str], tuple[_AcronymDefinition, ...]],
    conflict_blocks: frozenset[tuple[str, str]],
) -> PairDecision:
    if left.entity_type != right.entity_type:
        return _rejected(
            left,
            right,
            "incompatible_entity_types",
            "ontology entity types are incompatible",
        )
    if left.identifier and right.identifier:
        if left.identifier != right.identifier:
            return _rejected(
                left,
                right,
                "conflicting_stable_identifiers",
                "stable identifiers conflict",
            )
    version_signatures_differ = left.version_signature != right.version_signature
    if version_signatures_differ and (
        left.base_key == right.base_key
        or bool(left.identifier and left.identifier == right.identifier)
    ):
        return _rejected(
            left,
            right,
            "version_mismatch",
            "version signatures differ",
        )
    if left.identifier and left.identifier == right.identifier:
        return _accepted(
            left, right, "stable_identifier", "exact stable identifiers agree"
        )
    if left.is_pronoun or right.is_pronoun:
        return _rejected(
            left,
            right,
            "pronoun_only",
            "pronoun-only references are not resolved in version one",
        )
    name_block = (left.entity_type, left.normalized_label)
    if (
        left.normalized_label == right.normalized_label
        and name_block in conflict_blocks
    ):
        return _rejected(
            left,
            right,
            "conflicting_stable_identifiers",
            "same-name block contains conflicting stable identifiers",
        )
    acronym_decision = _acronym_decision(left, right, expansions)
    if acronym_decision is not None:
        return acronym_decision
    if left.normalized_label == right.normalized_label:
        if left.raw_entity_type != right.raw_entity_type:
            return _accepted(
                left,
                right,
                "ontology_alias",
                "same normalized name uses ontology-declared type aliases",
            )
        return _accepted(
            left, right, "normalized_name", "normalized names and types are identical"
        )
    return _rejected(
        left,
        right,
        "normalized_name_mismatch",
        "no conservative identity rule matched",
    )


def _constrain_component_merges(
    mentions: tuple[_MentionView, ...],
    decisions: tuple[PairDecision, ...],
) -> tuple[tuple[PairDecision, ...], tuple[PairDecision, ...]]:
    """Accept candidate edges only when their complete components are compatible."""

    indexes = {mention.mention_id: index for index, mention in enumerate(mentions)}
    decision_indexes = {
        frozenset((decision.left_mention_id, decision.right_mention_id)): index
        for index, decision in enumerate(decisions)
    }
    hard_conflicts: dict[int, dict[int, PairDecision]] = defaultdict(dict)
    for decision in decisions:
        if decision.accepted or decision.method not in _HARD_CANNOT_LINK_METHODS:
            continue
        left_index = indexes[decision.left_mention_id]
        right_index = indexes[decision.right_mention_id]
        hard_conflicts[left_index][right_index] = decision
        hard_conflicts[right_index][left_index] = decision

    candidates = sorted(
        (decision for decision in decisions if decision.accepted),
        key=lambda decision: (
            _METHOD_PRECEDENCE[decision.method],
            min(
                indexes[decision.left_mention_id],
                indexes[decision.right_mention_id],
            ),
            max(
                indexes[decision.left_mention_id],
                indexes[decision.right_mention_id],
            ),
        ),
    )
    constrained = list(decisions)
    disjoint_set = _DisjointSet(len(mentions))
    component_members: dict[int, set[int]] = {
        index: {index} for index in range(len(mentions))
    }
    merge_edges: list[PairDecision] = []
    for candidate in candidates:
        left_root = disjoint_set.find(indexes[candidate.left_mention_id])
        right_root = disjoint_set.find(indexes[candidate.right_mention_id])
        if left_root == right_root:
            continue
        left_members = component_members[left_root]
        right_members = component_members[right_root]
        blockers = [
            hard_conflicts[left_index][right_index]
            for left_index in sorted(left_members)
            for right_index in sorted(right_members)
            if right_index in hard_conflicts[left_index]
        ]
        if blockers:
            blocker = min(
                blockers,
                key=lambda decision: (
                    decision.method,
                    min(
                        indexes[decision.left_mention_id],
                        indexes[decision.right_mention_id],
                    ),
                    max(
                        indexes[decision.left_mention_id],
                        indexes[decision.right_mention_id],
                    ),
                ),
            )
            suppressed = PairDecision(
                left_mention_id=candidate.left_mention_id,
                right_mention_id=candidate.right_mention_id,
                accepted=False,
                method="component_conflict",
                confidence=0.0,
                explanation=(
                    "Rejected: component merge would violate "
                    f"{blocker.method} between {blocker.left_mention_id} and "
                    f"{blocker.right_mention_id}."
                ),
            )
            pair_key = frozenset(
                (candidate.left_mention_id, candidate.right_mention_id)
            )
            constrained[decision_indexes[pair_key]] = suppressed
            continue
        combined = left_members | right_members
        new_root = disjoint_set.union(left_root, right_root)
        component_members.pop(left_root)
        component_members.pop(right_root)
        component_members[new_root] = combined
        merge_edges.append(candidate)
    return tuple(constrained), tuple(merge_edges)


def _cluster_key(
    *,
    mentions: tuple[_MentionView, ...],
    ontology_checksum: str,
    entity_type: str,
    normalized_label: str,
    version_signature: str,
    identifier: str,
) -> str:
    value = {
        "resolver_version": DOCUMENT_RESOLVER_VERSION,
        "ontology_checksum": ontology_checksum,
        "document_ids": sorted({mention.document_id for mention in mentions}),
        "member_keys": sorted(mention.member_key for mention in mentions),
        "entity_type": entity_type,
        "normalized_label": normalized_label,
        "version_signature": version_signature,
        "identifier": identifier,
    }
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


def _build_clusters(
    mentions: tuple[_MentionView, ...],
    merge_edges: tuple[PairDecision, ...],
    *,
    ontology_checksum: str,
) -> tuple[ResolvedCluster, ...]:
    disjoint_set = _DisjointSet(len(mentions))
    indexes = {mention.mention_id: index for index, mention in enumerate(mentions)}
    for decision in merge_edges:
        disjoint_set.union(
            indexes[decision.left_mention_id], indexes[decision.right_mention_id]
        )
    groups: dict[int, list[_MentionView]] = defaultdict(list)
    for index, mention in enumerate(mentions):
        groups[disjoint_set.find(index)].append(mention)
    clusters: list[ResolvedCluster] = []
    for group in groups.values():
        ordered = tuple(sorted(group, key=lambda item: item.sort_key))
        representative = min(
            ordered,
            key=lambda item: (
                item.is_acronym or item.is_pronoun,
                -len(item.display_label),
                item.member_key,
                item.display_label,
            ),
        )
        member_ids = tuple(item.mention_id for item in ordered)
        identifiers = sorted({item.identifier for item in ordered if item.identifier})
        if len(identifiers) > 1:
            raise ValueError("resolved cluster contains conflicting stable identifiers")
        version_signatures = sorted(
            {
                item.version_signature
                for item in ordered
                if item.version_signature is not None
            }
        )
        if len(version_signatures) > 1:
            raise ValueError("resolved cluster contains conflicting version signatures")
        member_set = set(member_ids)
        methods = {
            decision.method
            for decision in merge_edges
            if decision.left_mention_id in member_set
            and decision.right_mention_id in member_set
        }
        method = min(methods or {"singleton"}, key=_METHOD_PRECEDENCE.__getitem__)
        confidence = min(item.confidence for item in ordered)
        adjacency: dict[str, list[tuple[str, PairDecision]]] = defaultdict(list)
        for decision in merge_edges:
            if (
                decision.left_mention_id in member_set
                and decision.right_mention_id in member_set
            ):
                adjacency[decision.left_mention_id].append(
                    (decision.right_mention_id, decision)
                )
                adjacency[decision.right_mention_id].append(
                    (decision.left_mention_id, decision)
                )
        mention_by_id = {item.mention_id: item for item in ordered}
        singleton = len(ordered) == 1
        membership_by_id = {
            representative.mention_id: ClusterMembership(
                mention_id=representative.mention_id,
                method="singleton" if singleton else "root",
                reason=(
                    "Singleton cluster." if singleton else "Deterministic cluster root."
                ),
                parent_mention_id=None,
            )
        }
        queue = [representative.mention_id]
        for parent_id in queue:
            for child_id, edge in sorted(
                adjacency[parent_id],
                key=lambda item: mention_by_id[item[0]].sort_key,
            ):
                if child_id in membership_by_id:
                    continue
                membership_by_id[child_id] = ClusterMembership(
                    mention_id=child_id,
                    method=edge.method,
                    reason=edge.explanation,
                    parent_mention_id=parent_id,
                )
                queue.append(child_id)
        if set(membership_by_id) != member_set:
            raise ValueError("cluster merge edges do not form a spanning tree")
        memberships = tuple(membership_by_id[mention.mention_id] for mention in ordered)
        version_signature = version_signatures[0] if version_signatures else ""
        clusters.append(
            ResolvedCluster(
                cluster_key=_cluster_key(
                    mentions=ordered,
                    ontology_checksum=ontology_checksum,
                    entity_type=representative.entity_type,
                    normalized_label=representative.normalized_label,
                    version_signature=version_signature,
                    identifier=identifiers[0] if identifiers else "",
                ),
                mention_ids=member_ids,
                memberships=memberships,
                label=representative.display_label,
                normalized_label=representative.normalized_label,
                version_signature=version_signature,
                entity_type=representative.entity_type,
                identifier=identifiers[0] if identifiers else "",
                method=method,
                confidence=confidence,
            )
        )
    return tuple(sorted(clusters, key=lambda item: item.cluster_key))


def _result_payload(
    *,
    resolver_version: str,
    ontology_checksum: str,
    input_fingerprint: str,
    mention_ids: tuple[str, ...],
    clusters: tuple[ResolvedCluster, ...],
    decisions: tuple[PairDecision, ...],
) -> dict[str, object]:
    return {
        "resolver_version": resolver_version,
        "ontology_checksum": ontology_checksum,
        "input_fingerprint": input_fingerprint,
        "mention_ids": mention_ids,
        "clusters": [asdict(cluster) for cluster in clusters],
        "decisions": [asdict(decision) for decision in decisions],
    }


def _payload_checksum(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def resolution_result_checksum(result: ResolutionResult) -> str:
    """Recompute a result digest before crossing a persistence boundary."""

    return _payload_checksum(
        _result_payload(
            resolver_version=result.resolver_version,
            ontology_checksum=result.ontology_checksum,
            input_fingerprint=result.input_fingerprint,
            mention_ids=result.mention_ids,
            clusters=result.clusters,
            decisions=result.decisions,
        )
    )


def resolve_document_mentions(
    mentions: Iterable[object], ontology: object
) -> ResolutionResult:
    """Partition one bounded document's mentions using deterministic exact rules."""

    try:
        bounded = tuple(islice(iter(mentions), MAX_DOCUMENT_MENTIONS + 1))
    except TypeError as exc:
        raise ValueError("mentions must be iterable") from exc
    if len(bounded) > MAX_DOCUMENT_MENTIONS:
        raise ValueError(f"document mention cap exceeded ({MAX_DOCUMENT_MENTIONS})")
    input_fingerprint = resolution_input_fingerprint(bounded)
    type_index, ontology_checksum = _ontology_type_index(ontology)
    adapted = tuple(
        sorted(
            (_adapt_mention(item, type_index) for item in bounded),
            key=lambda item: item.sort_key,
        )
    )
    mention_ids = tuple(item.mention_id for item in adapted)
    if len(set(mention_ids)) != len(mention_ids):
        raise ValueError("mention IDs must be unique within a document")
    member_keys = tuple(item.member_key for item in adapted)
    if len(set(member_keys)) != len(member_keys):
        raise ValueError("source-coordinate member identities must be unique")
    expansions = _acronym_expansions(adapted)
    conflict_blocks = _name_identifier_conflicts(adapted)
    candidate_decisions = tuple(
        _decide_pair(
            left,
            right,
            expansions=expansions,
            conflict_blocks=conflict_blocks,
        )
        for left, right in combinations(adapted, 2)
    )
    decisions, merge_edges = _constrain_component_merges(adapted, candidate_decisions)
    clusters = _build_clusters(
        adapted,
        merge_edges,
        ontology_checksum=ontology_checksum,
    )
    payload = _result_payload(
        resolver_version=DOCUMENT_RESOLVER_VERSION,
        ontology_checksum=ontology_checksum,
        input_fingerprint=input_fingerprint,
        mention_ids=mention_ids,
        clusters=clusters,
        decisions=decisions,
    )
    return ResolutionResult(
        resolver_version=DOCUMENT_RESOLVER_VERSION,
        ontology_checksum=ontology_checksum,
        input_fingerprint=input_fingerprint,
        mention_ids=mention_ids,
        clusters=clusters,
        decisions=decisions,
        checksum=_payload_checksum(payload),
    )


__all__ = [
    "MAX_DOCUMENT_MENTIONS",
    "ClusterMembership",
    "DocumentMention",
    "PairDecision",
    "ResolutionResult",
    "ResolvedCluster",
    "resolve_document_mentions",
    "resolution_input_fingerprint",
    "resolution_result_checksum",
]
