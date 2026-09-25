"""Pure, bounded, conservative within-document mention clustering."""

from __future__ import annotations

import json
import re
import unicodedata
from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from hashlib import sha256
from itertools import combinations, islice

from apps.knowledge_graph.extraction.pipeline import (
    ExtractionCapacityCode,
    ExtractionCapacityError,
)
from apps.knowledge_graph.extraction.windows import sanitize_graph_source_text

from . import DOCUMENT_RESOLVER_VERSION
from .coreference_types import (
    _HASH,
    ClusterMembership,
    DocumentMention,
    PairDecision,
    ResolvedCluster,
    _MentionView,
)
from .coreference_validation import (
    _MAX_ENTITY_TYPE_CHARACTERS,
    _MAX_UNIQUE_SOURCE_CONTEXT_CHARACTERS,
    _canonical_uuid,
    _confidence,
    _contains_unsafe_control,
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
from .normalization import normalize_entity_label, parse_stable_identifier

MAX_DOCUMENT_MENTIONS = 65_536
MAX_DOCUMENT_DECISIONS = 524_288
_EXHAUSTIVE_PAIR_LIMIT = 512
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
        if type(self.ontology_checksum) is not str or not _HASH.fullmatch(
            self.ontology_checksum
        ):
            raise ValueError("ontology_checksum must be a lowercase SHA-256 digest")
        if type(self.input_fingerprint) is not str or not _HASH.fullmatch(
            self.input_fingerprint
        ):
            raise ValueError("input_fingerprint must be a lowercase SHA-256 digest")
        if type(self.mention_ids) is not tuple:
            raise ValueError("mention_ids must be a tuple")
        if not all(type(mention_id) is str for mention_id in self.mention_ids):
            raise ValueError("mention_ids must contain exact strings")
        if len(set(self.mention_ids)) != len(self.mention_ids):
            raise ValueError("result mention IDs must be unique")
        if type(self.clusters) is not tuple or not all(
            isinstance(cluster, ResolvedCluster) for cluster in self.clusters
        ):
            raise ValueError("clusters must contain ResolvedCluster values")
        cluster_keys = tuple(cluster.cluster_key for cluster in self.clusters)
        if len(set(cluster_keys)) != len(cluster_keys):
            raise ValueError("cluster keys must be unique")
        if type(self.decisions) is not tuple or not all(
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
        mention_id_set = set(self.mention_ids)
        actual_pairs = {
            frozenset((decision.left_mention_id, decision.right_mention_id))
            for decision in self.decisions
        }
        if (
            len(self.decisions) > MAX_DOCUMENT_DECISIONS
            or len(actual_pairs) != len(self.decisions)
            or any(
                decision.left_mention_id == decision.right_mention_id
                or decision.left_mention_id not in mention_id_set
                or decision.right_mention_id not in mention_id_set
                for decision in self.decisions
            )
        ):
            raise ValueError("decisions must be unique bounded mention pairs")
        if len(self.mention_ids) <= _EXHAUSTIVE_PAIR_LIMIT:
            expected_pairs = {
                frozenset(pair) for pair in combinations(self.mention_ids, 2)
            }
            if actual_pairs != expected_pairs:
                raise ValueError("decisions must audit every mention pair exactly once")
        accepted_pairs = {
            frozenset((decision.left_mention_id, decision.right_mention_id))
            for decision in self.decisions
            if decision.accepted
        }
        membership_edges = {
            frozenset((membership.mention_id, membership.parent_mention_id))
            for cluster in self.clusters
            for membership in cluster.memberships
            if membership.parent_mention_id is not None
        }
        if not membership_edges.issubset(accepted_pairs):
            raise ValueError("membership parent edges require accepted decisions")
        if type(self.checksum) is not str or not _HASH.fullmatch(self.checksum):
            raise ValueError("checksum must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class _PreparedSource:
    document_id: str
    source_key: str
    coordinate_scope: str
    chunk_id: str
    position_basis: str
    content_object_id: str
    source_text: str
    source_offset: int
    context_digest: str


@dataclass(frozen=True, slots=True)
class _AcronymDefinition:
    expansion: str
    coordinate_scope: str
    full_mention_id: str
    acronym_mention_id: str
    definition_start: int


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self._parents = list(range(size))
        self._sizes = [1] * size

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
        if self._sizes[left_root] < self._sizes[right_root] or (
            self._sizes[left_root] == self._sizes[right_root]
            and left_root > right_root
        ):
            left_root, right_root = right_root, left_root
        self._parents[right_root] = left_root
        self._sizes[left_root] += self._sizes[right_root]
        return left_root


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
    if type(persisted_checksum) is not str:
        raise ValueError("ontology checksum must be a lowercase SHA-256 digest")
    if persisted_checksum != "":
        if not _HASH.fullmatch(persisted_checksum):
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


def _metadata_identifier(mention: object) -> str:
    direct = _value(mention, "identifier", "")
    candidates: list[tuple[str, str, str]] = []
    if type(direct) is not str:
        raise ValueError("identifier must be a string")
    if direct != "":
        validated = _validated_identifier(direct)
        parsed = parse_stable_identifier(validated)
        if parsed is None:
            raise ValueError("explicit identifier must be a valid stable identifier")
        candidates.append(("identifier", validated, parsed.canonical))
    metadata = _value(mention, "metadata", {})
    if isinstance(metadata, Mapping):
        for key in ("stable_identifier", "identifier"):
            if key not in metadata:
                continue
            value = metadata[key]
            if type(value) is not str or not value.strip():
                raise ValueError(f"metadata {key} identifier must be nonempty text")
            validated = _validated_identifier(value)
            parsed = parse_stable_identifier(validated)
            if parsed is None:
                raise ValueError(
                    f"metadata {key} identifier must be a valid stable identifier"
                )
            candidates.append((key, validated, parsed.canonical))
    canonical_values = {canonical for _, _, canonical in candidates}
    if len(canonical_values) > 1:
        raise ValueError("identifier fields contain conflicting stable identifiers")
    return candidates[0][1] if candidates else ""


def _raw_source_context(mention: object) -> tuple[object, int]:
    explicit_text = _value(mention, "source_text", _MISSING)
    explicit_offset = _value(mention, "source_offset", _MISSING)
    validated_explicit_offset = (
        0
        if explicit_offset is _MISSING
        else _validated_db_integer(
            explicit_offset,
            "source_offset",
            minimum=0,
        )
    )
    if explicit_text is not _MISSING:
        return explicit_text, validated_explicit_offset
    chunk = _value(mention, "chunk")
    if chunk is None:
        return "", validated_explicit_offset
    content = _value(chunk, "content", "")
    if type(content) is str:
        content = sanitize_graph_source_text(content)
    basis = _value(mention, "position_basis", "document_global")
    if explicit_offset is not _MISSING:
        return content, validated_explicit_offset
    offset = _value(chunk, "start_position", 0) if basis == "document_global" else 0
    validated_offset = _validated_db_integer(offset, "source_offset", minimum=0)
    return content, validated_offset


def _source_identity(
    mention: object,
    *,
    source_offset: int,
) -> tuple[str, str, str, str, str, str]:
    document_id = _canonical_uuid(_value(mention, "document_id"), "document_id")
    chunk_id = _validated_db_integer(_value(mention, "chunk_id"), "chunk_id", minimum=1)
    position_basis, content_object_id = _validated_coordinate_basis(
        _value(mention, "position_basis", "document_global"),
        _value(mention, "content_object_id"),
    )
    explicit_source_key = _value(mention, "source_key", _MISSING)
    if explicit_source_key is not _MISSING and type(explicit_source_key) is not str:
        raise ValueError("source key must be a nonempty string")
    if explicit_source_key is not _MISSING and explicit_source_key != "":
        source_key = _validated_source_key(explicit_source_key)
    elif position_basis == "document_global":
        source_key = f"document:{document_id}:offset:{source_offset}"
    elif content_object_id:
        source_key = f"content:{content_object_id}"
    coordinate_scope = (
        f"document:{document_id}"
        if position_basis == "document_global"
        else f"content:{content_object_id}"
    )
    return (
        document_id,
        _validated_source_key(source_key),
        coordinate_scope,
        str(chunk_id),
        position_basis,
        content_object_id,
    )


def _prepare_source_contexts(
    mentions: tuple[object, ...],
) -> tuple[_PreparedSource, ...]:
    cached_contexts: dict[str, tuple[str, str]] = {}
    prepared: list[_PreparedSource] = []
    aggregate_source_characters = 0
    document_id_seen: str | None = None
    for mention in mentions:
        raw_source_text, source_offset = _raw_source_context(mention)
        (
            document_id,
            source_key,
            coordinate_scope,
            chunk_id,
            position_basis,
            content_object_id,
        ) = _source_identity(mention, source_offset=source_offset)
        if document_id_seen is None:
            document_id_seen = document_id
        elif document_id != document_id_seen:
            raise ValueError("mentions must belong to a single document")
        cached = cached_contexts.get(source_key)
        if cached is not None:
            source_text, context_digest = cached
            if type(raw_source_text) is not str:
                raise ValueError("source text must be a string")
            if raw_source_text != source_text:
                raise ValueError("source key has mismatched source context text")
        else:
            source_text = _validated_source_text(raw_source_text)
            aggregate_source_characters += len(source_text)
            if aggregate_source_characters > _MAX_UNIQUE_SOURCE_CONTEXT_CHARACTERS:
                raise ExtractionCapacityError(
                    ExtractionCapacityCode.CHARACTER_LIMIT,
                    "aggregate unique source context exceeds the "
                    f"{_MAX_UNIQUE_SOURCE_CONTEXT_CHARACTERS}-character limit"
                )
            context_digest = _source_context_digest(
                source_key=source_key,
                source_text=source_text,
            )
            cached_contexts[source_key] = (source_text, context_digest)
        prepared.append(
            _PreparedSource(
                document_id=document_id,
                source_key=source_key,
                coordinate_scope=coordinate_scope,
                chunk_id=chunk_id,
                position_basis=position_basis,
                content_object_id=content_object_id,
                source_text=source_text,
                source_offset=source_offset,
                context_digest=context_digest,
            )
        )
    return tuple(prepared)


def _member_key(
    *,
    document_id: str,
    coordinate_scope: str,
    position_basis: str,
    content_object_id: str,
    start: int,
    end: int,
    entity_type: str,
    normalized_label: str,
) -> str:
    payload = {
        "document_id": document_id,
        "coordinate_scope": coordinate_scope,
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


def _adapt_mention(
    mention: object,
    type_index: Mapping[str, str],
    source: _PreparedSource,
) -> _MentionView:
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
    start, end = _validated_span(start, end)
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
        source_text=source.source_text,
        source_offset=source.source_offset,
        document_id=source.document_id,
        source_key=source.source_key,
        coordinate_scope=source.coordinate_scope,
        chunk_id=source.chunk_id,
        position_basis=source.position_basis,
        content_object_id=source.content_object_id,
        member_key=_member_key(
            document_id=source.document_id,
            coordinate_scope=source.coordinate_scope,
            position_basis=source.position_basis,
            content_object_id=source.content_object_id,
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


def _resolution_input_fingerprint(
    mentions: tuple[object, ...],
    sources: tuple[_PreparedSource, ...],
) -> str:
    records: list[dict[str, object]] = []
    source_contexts: dict[tuple[str, str], dict[str, object]] = {}
    for mention, source in zip(mentions, sources, strict=True):
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
        start, end = _validated_span(start, end)
        context_key = (source.source_key, source.context_digest)
        source_contexts.setdefault(
            context_key,
            {
                "digest": source.context_digest,
                "source_key": source.source_key,
                "character_count": len(source.source_text),
            },
        )
        records.append(
            {
                "mention_id": _mention_key(mention_id),
                "document_id": source.document_id,
                "source_key": source.source_key,
                "chunk_id": source.chunk_id,
                "position_basis": source.position_basis,
                "content_object_id": source.content_object_id,
                "start": start,
                "end": end,
                "raw_text": validated_raw_text,
                "entity_type": validated_entity_type,
                "identifier": str(_metadata_identifier(mention) or ""),
                "confidence": _confidence(confidence),
                "source_offset": source.source_offset,
                "source_context_digest": source.context_digest,
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


def resolution_input_fingerprint(mentions: Iterable[object]) -> str:
    """Bind a result to exact mention fields and local source context."""

    bounded = tuple(islice(iter(mentions), MAX_DOCUMENT_MENTIONS + 1))
    if len(bounded) > MAX_DOCUMENT_MENTIONS:
        raise ValueError(f"document mention cap exceeded ({MAX_DOCUMENT_MENTIONS})")
    sources = _prepare_source_contexts(bounded)
    return _resolution_input_fingerprint(bounded, sources)


def _source_context_digest(*, source_key: str, source_text: str) -> str:
    digest = sha256()
    for value in (source_key, source_text):
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
        full.coordinate_scope != acronym.coordinate_scope
        or full.end >= acronym.start
        or _initialism(full.display_label) != _acronym_key(acronym.display_label)
    ):
        return False
    for source, source_offset in (
        (full.source_text, full.source_offset),
        (acronym.source_text, acronym.source_offset),
    ):
        full_end = full.end - source_offset
        acronym_start = acronym.start - source_offset
        acronym_end = acronym.end - source_offset
        if not (0 <= full_end <= acronym_start < acronym_end <= len(source)):
            continue
        between = source[full_end:acronym_start]
        following = source[acronym_end:]
        if re.fullmatch(r"\s*\(\s*", between) and re.match(r"\s*\)", following):
            return True
    return False


def _acronym_expansions(
    mentions: tuple[_MentionView, ...],
) -> dict[tuple[str, str], tuple[_AcronymDefinition, ...]]:
    definitions: dict[tuple[str, str], list[_AcronymDefinition]] = defaultdict(list)
    full_by_key: dict[tuple[str, str, str], list[_MentionView]] = defaultdict(list)
    contexts: dict[tuple[str, str, int], str] = {}
    acronym_positions = {
        (mention.coordinate_scope, mention.start)
        for mention in mentions
        if mention.is_acronym
    }
    for mention in mentions:
        contexts.setdefault(
            (mention.coordinate_scope, mention.source_key, mention.source_offset),
            mention.source_text,
        )
        if mention.is_acronym:
            continue
        initialism = _initialism(mention.display_label)
        if initialism:
            full_by_key[
                (mention.entity_type, initialism, mention.coordinate_scope)
            ].append(mention)
    for group in full_by_key.values():
        group.sort(key=lambda item: (item.end, item.sort_key))
    full_ends_by_key = {
        key: tuple(full.end for full in group)
        for key, group in full_by_key.items()
    }
    parenthetical_intervals: dict[
        tuple[str, int], set[tuple[int, int]]
    ] = defaultdict(set)
    for (
        coordinate_scope,
        _source_key,
        source_offset,
    ), source_text in contexts.items():
        for opening_index, character in enumerate(source_text):
            if character != "(":
                continue
            full_end = opening_index
            while full_end > 0 and source_text[full_end - 1].isspace():
                full_end -= 1
            acronym_start = opening_index + 1
            while (
                acronym_start < len(source_text)
                and source_text[acronym_start].isspace()
            ):
                acronym_start += 1
            position_key = (coordinate_scope, source_offset + acronym_start)
            if position_key in acronym_positions:
                parenthetical_intervals[position_key].add(
                    (source_offset + full_end, source_offset + opening_index)
                )
    candidate_checks = 0
    for acronym in mentions:
        if not acronym.is_acronym:
            continue
        full_key = (
            acronym.entity_type,
            _acronym_key(acronym.display_label),
            acronym.coordinate_scope,
        )
        full_group = full_by_key.get(full_key, ())
        full_ends = full_ends_by_key.get(full_key, ())
        candidates_by_id: dict[str, _MentionView] = {}
        for minimum_end, maximum_end in sorted(
            parenthetical_intervals.get(
                (acronym.coordinate_scope, acronym.start),
                (),
            )
        ):
            start_index = bisect_left(full_ends, minimum_end)
            stop_index = bisect_right(full_ends, maximum_end)
            for full in full_group[start_index:stop_index]:
                candidates_by_id[full.mention_id] = full
        candidates = tuple(
            sorted(candidates_by_id.values(), key=lambda item: item.sort_key)
        )
        candidate_checks += len(candidates)
        if candidate_checks > MAX_DOCUMENT_DECISIONS:
            raise ExtractionCapacityError(
                ExtractionCapacityCode.ENTITY_LIMIT,
                "document acronym definition audit cap exceeded",
            )
        for full in candidates:
            if _is_parenthetical_definition(full, acronym):
                definitions[
                    (acronym.entity_type, _acronym_key(acronym.display_label))
                ].append(
                    _AcronymDefinition(
                        expansion=full.normalized_label,
                        coordinate_scope=full.coordinate_scope,
                        full_mention_id=full.mention_id,
                        acronym_mention_id=acronym.mention_id,
                        definition_start=acronym.start,
                    )
                )
    result: dict[tuple[str, str], tuple[_AcronymDefinition, ...]] = {}
    for key, values in definitions.items():
        earliest: dict[tuple[str, str], _AcronymDefinition] = {}
        for value in values:
            definition_key = (value.expansion, value.coordinate_scope)
            existing = earliest.get(definition_key)
            if existing is None or (
                value.definition_start,
                value.full_mention_id,
                value.acronym_mention_id,
            ) < (
                existing.definition_start,
                existing.full_mention_id,
                existing.acronym_mention_id,
            ):
                earliest[definition_key] = value
        result[key] = tuple(
            sorted(
                earliest.values(),
                key=lambda item: (
                    item.coordinate_scope,
                    item.definition_start,
                    item.full_mention_id,
                    item.acronym_mention_id,
                ),
            )
        )
    return result


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


def _sparse_candidate_decisions(
    mentions: tuple[_MentionView, ...],
    *,
    expansions: Mapping[tuple[str, str], tuple[_AcronymDefinition, ...]],
    conflict_blocks: frozenset[tuple[str, str]],
) -> tuple[PairDecision, ...]:
    """Audit only pairs that can merge or conservatively block a merge."""

    indexes = {mention.mention_id: index for index, mention in enumerate(mentions)}
    pair_indexes: set[tuple[int, int]] = set()

    def add_pair(left: _MentionView, right: _MentionView) -> None:
        left_index = indexes[left.mention_id]
        right_index = indexes[right.mention_id]
        pair = tuple(sorted((left_index, right_index)))
        if pair[0] == pair[1] or pair in pair_indexes:
            return
        if len(pair_indexes) >= MAX_DOCUMENT_DECISIONS:
            raise ExtractionCapacityError(
                ExtractionCapacityCode.ENTITY_LIMIT,
                "document coreference candidate audit cap exceeded",
            )
        pair_indexes.add(pair)

    def add_star(
        group: list[_MentionView], *, anchor: _MentionView | None = None
    ) -> None:
        if len(group) < 2:
            return
        anchor = anchor or group[0]
        for mention in group:
            if mention is anchor:
                continue
            add_pair(anchor, mention)

    def add_complete(group: list[_MentionView]) -> None:
        for left, right in combinations(group, 2):
            add_pair(left, right)

    identifiers: dict[tuple[str, str], list[_MentionView]] = defaultdict(list)
    names: dict[tuple[str, str], list[_MentionView]] = defaultdict(list)
    entity_types: dict[str, list[_MentionView]] = defaultdict(list)
    acronym_keys: set[tuple[str, str]] = set()
    for mention in mentions:
        if mention.identifier:
            identifiers[(mention.entity_type, mention.identifier)].append(mention)
        names[(mention.entity_type, mention.normalized_label)].append(mention)
        entity_types[mention.entity_type].append(mention)
        if mention.is_acronym:
            acronym_keys.add((mention.entity_type, mention.acronym_shape_key))

    def has_local_acronym_definition(mention: _MentionView) -> bool:
        return any(
            definition.coordinate_scope == mention.coordinate_scope
            for definition in expansions.get(
                (mention.entity_type, mention.acronym_shape_key),
                (),
            )
        )

    for group in identifiers.values():
        signatures = {mention.version_signature for mention in group}
        if len(signatures) > 1:
            add_complete(group)
        else:
            add_star(group)

    for name_key, group in names.items():
        is_acronym_block = any(
            mention.is_acronym
            or has_local_acronym_definition(mention)
            for mention in group
        )
        if is_acronym_block:
            continue
        requires_complete_audit = (
            name_key in conflict_blocks
            or any(mention.is_pronoun for mention in group)
        )
        if requires_complete_audit:
            add_complete(group)
        else:
            unidentified = next(
                (mention for mention in group if not mention.identifier),
                group[0],
            )
            add_star(group, anchor=unidentified)

    for group in entity_types.values():
        global_blockers = [
            mention
            for mention in group
            if mention.is_pronoun
            or (
                not mention.is_acronym
                and has_local_acronym_definition(mention)
            )
        ]
        for blocker in global_blockers:
            for mention in group:
                add_pair(blocker, mention)

    acronym_groups: dict[tuple[str, str], list[_MentionView]] = defaultdict(list)
    for mention in mentions:
        if (mention.entity_type, mention.acronym_shape_key) in acronym_keys:
            acronym_groups[(mention.entity_type, mention.acronym_shape_key)].append(
                mention
            )
        initialism = _initialism(mention.display_label)
        if (mention.entity_type, initialism) in acronym_keys:
            acronym_groups[(mention.entity_type, initialism)].append(mention)
    for group in acronym_groups.values():
        unique_group = list(dict.fromkeys(group))
        if len(unique_group) < 2:
            continue
        equivalent_anchor = next(
            (mention for mention in unique_group if not mention.is_acronym),
            unique_group[0],
        )
        anchor_decisions = tuple(
            _decide_pair(
                equivalent_anchor,
                mention,
                expansions=expansions,
                conflict_blocks=conflict_blocks,
            )
            for mention in unique_group
            if mention is not equivalent_anchor
        )
        identifiers_in_group = {
            mention.identifier for mention in unique_group if mention.identifier
        }
        versions_in_group = {
            mention.version_signature
            for mention in unique_group
            if mention.version_signature is not None
        }
        if (
            len(identifiers_in_group) <= 1
            and len(versions_in_group) <= 1
            and anchor_decisions
            and all(decision.accepted for decision in anchor_decisions)
        ):
            add_star(unique_group, anchor=equivalent_anchor)
        else:
            add_complete(unique_group)

    return tuple(
        _decide_pair(
            mentions[left_index],
            mentions[right_index],
            expansions=expansions,
            conflict_blocks=conflict_blocks,
        )
        for left_index, right_index in sorted(pair_indexes)
    )


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
    left_definitions = tuple(
        definition
        for definition in expansions.get((left.entity_type, left.acronym_shape_key), ())
        if definition.coordinate_scope == left.coordinate_scope
    )
    right_definitions = tuple(
        definition
        for definition in expansions.get(
            (right.entity_type, right.acronym_shape_key), ()
        )
        if definition.coordinate_scope == right.coordinate_scope
    )
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
        if left.coordinate_scope != right.coordinate_scope:
            return _rejected(
                left,
                right,
                "source_mismatch",
                "acronym definition belongs to another source coordinate space",
            )
        definitions = left_definitions
        candidate_expansions = {item.expansion for item in definitions}
        if len(candidate_expansions) > 1:
            return _rejected(left, right, "ambiguous_acronym", "ambiguous acronym")
        if not definitions:
            return _rejected(left, right, "undefined_acronym", "undefined acronym")

        def is_resolved(mention: _MentionView) -> bool:
            return any(
                definition.coordinate_scope == mention.coordinate_scope
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
    if acronym.coordinate_scope != full.coordinate_scope:
        return _rejected(
            left,
            right,
            "source_mismatch",
            "acronym definition belongs to another source coordinate space",
        )
    definitions = tuple(
        definition
        for definition in expansions.get((acronym.entity_type, acronym_key), ())
        if definition.coordinate_scope == acronym.coordinate_scope
    )
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
    if any(acronym.start >= item.definition_start for item in relevant):
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
    component_forbidden: dict[int, set[int]] = {
        index: set(hard_conflicts[index]) for index in range(len(mentions))
    }

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
        blocked_right = component_forbidden[left_root].intersection(right_members)
        blockers = [
            decision
            for right_index in blocked_right
            for left_index, decision in hard_conflicts[right_index].items()
            if left_index in left_members
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
        if len(left_members) < len(right_members):
            left_members, right_members = right_members, left_members
        left_members.update(right_members)
        left_forbidden = component_forbidden[left_root]
        right_forbidden = component_forbidden[right_root]
        if len(left_forbidden) < len(right_forbidden):
            left_forbidden, right_forbidden = right_forbidden, left_forbidden
        left_forbidden.update(right_forbidden)
        new_root = disjoint_set.union(left_root, right_root)
        component_members.pop(left_root)
        component_members.pop(right_root)
        component_forbidden.pop(left_root)
        component_forbidden.pop(right_root)
        component_members[new_root] = left_members
        component_forbidden[new_root] = left_forbidden
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
    edges_by_root: dict[int, list[PairDecision]] = defaultdict(list)
    for decision in merge_edges:
        edges_by_root[
            disjoint_set.find(indexes[decision.left_mention_id])
        ].append(decision)
    clusters: list[ResolvedCluster] = []
    for root, group in groups.items():
        ordered = tuple(sorted(group, key=lambda item: item.sort_key))
        cluster_edges = edges_by_root[root]
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
            for decision in cluster_edges
        }
        method = min(methods or {"singleton"}, key=_METHOD_PRECEDENCE.__getitem__)
        confidence = min(item.confidence for item in ordered)
        adjacency: dict[str, list[tuple[str, PairDecision]]] = defaultdict(list)
        for decision in cluster_edges:
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
    sources = _prepare_source_contexts(bounded)
    input_fingerprint = _resolution_input_fingerprint(bounded, sources)
    type_index, ontology_checksum = _ontology_type_index(ontology)
    adapted = tuple(
        sorted(
            (
                _adapt_mention(item, type_index, source)
                for item, source in zip(bounded, sources, strict=True)
            ),
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
    if len(adapted) <= _EXHAUSTIVE_PAIR_LIMIT:
        candidate_decisions = tuple(
            _decide_pair(
                left,
                right,
                expansions=expansions,
                conflict_blocks=conflict_blocks,
            )
            for left, right in combinations(adapted, 2)
        )
    else:
        candidate_decisions = _sparse_candidate_decisions(
            adapted,
            expansions=expansions,
            conflict_blocks=conflict_blocks,
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
    "MAX_DOCUMENT_DECISIONS",
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
