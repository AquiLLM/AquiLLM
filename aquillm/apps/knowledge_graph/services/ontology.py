"""Validated, provider-neutral research ontology definitions.

Checksums are SHA-256 digests of canonical JSON semantic content.  YAML key and
list ordering therefore never changes a definition's identity; ``raw_yaml`` is
retained separately for audit persistence.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from math import isfinite
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

_SEMVER = re.compile(
    r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
    r"(?:-(?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*)"
    r"(?:\.(?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*))*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)
_DIRECTIONS = frozenset({"directed", "undirected"})
_GRAPH_ONTOLOGY_ACTIVATION_LOCK = 707_750_921
_ENTITY_FIELDS = frozenset(
    {
        "name",
        "description",
        "aliases",
        "default_retrieval_weight",
        "default_suppression_policy",
        "default_suppression_threshold",
    }
)
_RELATION_FIELDS = frozenset(
    {
        "name",
        "description",
        "direction",
        "allowed_head_types",
        "allowed_tail_types",
    }
)


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that preserves duplicate-key errors instead of overwriting."""


def _construct_unique_mapping(
    loader: yaml.SafeLoader, node: yaml.nodes.MappingNode, deep: bool = False
) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise yaml.constructor.ConstructorError(
                None, None, "YAML mapping keys must be strings", key_node.start_mark
            )
        if key in mapping:
            raise yaml.constructor.ConstructorError(
                None,
                None,
                f"duplicate YAML mapping key: {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


class OntologyValidationError(ValueError):
    """Raised when an ontology document violates the stable schema."""


@dataclass(frozen=True, slots=True)
class EntityTypeDefinition:
    name: str
    description: str
    aliases: tuple[str, ...]
    default_retrieval_weight: float
    default_suppression_policy: str
    default_suppression_threshold: float


@dataclass(frozen=True, slots=True)
class RelationDefinition:
    name: str
    description: str
    direction: str
    allowed_head_types: tuple[str, ...]
    allowed_tail_types: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class OntologyDefinition:
    version: str
    entity_types: Mapping[str, EntityTypeDefinition]
    relations: Mapping[str, RelationDefinition]
    checksum: str
    canonical_yaml: str
    raw_yaml: str
    provenance: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True, slots=True)
class EntityTypeExtension:
    name: str
    description: str | None = None
    aliases: tuple[str, ...] | None = None
    default_retrieval_weight: float | None = None
    default_suppression_policy: str | None = None
    default_suppression_threshold: float | None = None


@dataclass(frozen=True, slots=True)
class RelationExtension:
    name: str
    description: str | None = None
    direction: str | None = None
    allowed_head_types: tuple[str, ...] | None = None
    allowed_tail_types: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class OntologyExtensionDefinition:
    version: str
    entity_types: Mapping[str, EntityTypeExtension]
    relations: Mapping[str, RelationExtension]
    checksum: str
    canonical_yaml: str
    raw_yaml: str


def _read_yaml(path: str | Path) -> tuple[Any, str]:
    source = Path(path).expanduser().resolve()
    try:
        raw_yaml = source.read_text(encoding="utf-8").replace("\r\n", "\n")
    except OSError as exc:
        raise OntologyValidationError(
            f"Unable to read ontology {source}: {exc}"
        ) from exc
    try:
        data = yaml.load(raw_yaml, Loader=_UniqueKeySafeLoader)
    except (ValueError, yaml.YAMLError) as exc:
        raise OntologyValidationError(f"Unsupported or malformed YAML: {exc}") from exc
    _validate_yaml_value(data)
    return data, raw_yaml


def _validate_yaml_value(value: Any, ancestors: set[int] | None = None) -> None:
    ancestors = set() if ancestors is None else ancestors
    if isinstance(value, Mapping):
        value_id = id(value)
        if value_id in ancestors:
            raise OntologyValidationError("YAML aliases must not create cycles")
        ancestors.add(value_id)
        for key, child in value.items():
            if not isinstance(key, str):
                raise OntologyValidationError("YAML mapping keys must be strings")
            _validate_yaml_value(child, ancestors)
        ancestors.remove(value_id)
    elif isinstance(value, list):
        value_id = id(value)
        if value_id in ancestors:
            raise OntologyValidationError("YAML aliases must not create cycles")
        ancestors.add(value_id)
        for child in value:
            _validate_yaml_value(child, ancestors)
        ancestors.remove(value_id)
    elif value is not None and not isinstance(value, (str, int, float, bool)):
        raise OntologyValidationError("YAML contains an unsupported value type")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OntologyValidationError(f"{label} must be a mapping")
    return value


def _require_fields(
    value: Mapping[str, Any], expected: frozenset[str], label: str
) -> None:
    unknown = set(value).difference(expected)
    if unknown:
        raise OntologyValidationError(
            f"{label} has unsupported fields: {sorted(unknown)}"
        )


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OntologyValidationError(f"{label} must be a nonempty string")
    return value.strip()


def _semantic_version(value: Any) -> str:
    version = _nonempty_string(value, "version")
    if len(version) > 128:
        raise OntologyValidationError("version must be at most 128 characters")
    if not _SEMVER.fullmatch(version):
        raise OntologyValidationError("version must be a semantic version")
    return version


def _compare_semver_precedence(left: str, right: str) -> int:
    """Compare validated SemVer values, intentionally ignoring build metadata."""
    left_core, _, left_prerelease = left.split("+", 1)[0].partition("-")
    right_core, _, right_prerelease = right.split("+", 1)[0].partition("-")
    left_numbers = tuple(int(part) for part in left_core.split("."))
    right_numbers = tuple(int(part) for part in right_core.split("."))
    if left_numbers != right_numbers:
        return 1 if left_numbers > right_numbers else -1
    if not left_prerelease or not right_prerelease:
        if left_prerelease == right_prerelease:
            return 0
        return -1 if left_prerelease else 1
    left_identifiers = left_prerelease.split(".")
    right_identifiers = right_prerelease.split(".")
    for left_identifier, right_identifier in zip(
        left_identifiers, right_identifiers, strict=False
    ):
        if left_identifier == right_identifier:
            continue
        left_numeric = left_identifier.isascii() and left_identifier.isdigit()
        right_numeric = right_identifier.isascii() and right_identifier.isdigit()
        if left_numeric and right_numeric:
            return 1 if int(left_identifier) > int(right_identifier) else -1
        if left_numeric != right_numeric:
            return -1 if left_numeric else 1
        return 1 if left_identifier > right_identifier else -1
    if len(left_identifiers) == len(right_identifiers):
        return 0
    return 1 if len(left_identifiers) > len(right_identifiers) else -1


def _names(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise OntologyValidationError(f"{label} must be a nonempty list")
    names = tuple(_nonempty_string(item, label) for item in value)
    if len(set(names)) != len(names):
        raise OntologyValidationError(f"{label} must not contain duplicate names")
    return tuple(sorted(names))


def _aliases(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise OntologyValidationError(f"{label} must be a list")
    aliases = tuple(_nonempty_string(item, label) for item in value)
    if len(set(aliases)) != len(aliases):
        raise OntologyValidationError(f"{label} must not contain duplicate aliases")
    return tuple(sorted(aliases))


def _unit_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OntologyValidationError(f"{label} must be a finite number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise OntologyValidationError(f"{label} must be between 0 and 1") from exc
    if not isfinite(number) or not 0.0 <= number <= 1.0:
        raise OntologyValidationError(f"{label} must be between 0 and 1")
    return number


def _records(value: Any, label: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        raise OntologyValidationError(f"{label} must be a list")
    return [_mapping(item, label) for item in value]


def _canonical_yaml(content: Mapping[str, Any]) -> str:
    return yaml.safe_dump(dict(content), allow_unicode=True, sort_keys=True)


def _checksum(content: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        content, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _definition_content(
    version: str,
    entity_types: Mapping[str, EntityTypeDefinition],
    relations: Mapping[str, RelationDefinition],
) -> dict[str, Any]:
    return {
        "version": version,
        "entity_types": [
            {
                "name": entity.name,
                "description": entity.description,
                "aliases": list(entity.aliases),
                "default_retrieval_weight": entity.default_retrieval_weight,
                "default_suppression_policy": entity.default_suppression_policy,
                "default_suppression_threshold": entity.default_suppression_threshold,
            }
            for entity in (entity_types[name] for name in sorted(entity_types))
        ],
        "relations": [
            {
                "name": relation.name,
                "description": relation.description,
                "direction": relation.direction,
                "allowed_head_types": list(relation.allowed_head_types),
                "allowed_tail_types": list(relation.allowed_tail_types),
            }
            for relation in (relations[name] for name in sorted(relations))
        ],
    }


def _build_definition(
    version: str,
    entity_types: Mapping[str, EntityTypeDefinition],
    relations: Mapping[str, RelationDefinition],
    raw_yaml: str | None,
    provenance: Mapping[str, str] | None = None,
) -> OntologyDefinition:
    content = _definition_content(version, entity_types, relations)
    canonical_yaml = _canonical_yaml(content)
    return OntologyDefinition(
        version=version,
        entity_types=MappingProxyType(dict(sorted(entity_types.items()))),
        relations=MappingProxyType(dict(sorted(relations.items()))),
        checksum=_checksum(content),
        canonical_yaml=canonical_yaml,
        raw_yaml=canonical_yaml if raw_yaml is None else raw_yaml,
        provenance=MappingProxyType(dict(sorted((provenance or {}).items()))),
    )


def load_ontology(path: str | Path) -> OntologyDefinition:
    """Load a complete ontology without importing Django or any LLM provider."""
    document, raw_yaml = _read_yaml(path)
    root = _mapping(document, "ontology")
    _require_fields(
        root, frozenset({"version", "entity_types", "relations"}), "ontology"
    )
    version = _semantic_version(root.get("version"))

    entity_types: dict[str, EntityTypeDefinition] = {}
    known_aliases: set[str] = set()
    for record in _records(root.get("entity_types"), "entity_types"):
        _require_fields(record, _ENTITY_FIELDS, "entity type")
        missing = _ENTITY_FIELDS.difference(record)
        if missing:
            raise OntologyValidationError(
                f"entity type is missing fields: {sorted(missing)}"
            )
        name = _nonempty_string(record["name"], "entity type name")
        aliases = _aliases(record["aliases"], f"aliases for {name}")
        if name in entity_types or name in known_aliases or name in aliases:
            raise OntologyValidationError(
                f"duplicate entity type name or alias: {name}"
            )
        if set(aliases).intersection(entity_types) or set(aliases).intersection(
            known_aliases
        ):
            raise OntologyValidationError(f"duplicate entity aliases for {name}")
        entity_types[name] = EntityTypeDefinition(
            name=name,
            description=_nonempty_string(
                record["description"], f"description for {name}"
            ),
            aliases=aliases,
            default_retrieval_weight=_unit_number(
                record["default_retrieval_weight"], f"retrieval weight for {name}"
            ),
            default_suppression_policy=_nonempty_string(
                record["default_suppression_policy"], f"suppression policy for {name}"
            ),
            default_suppression_threshold=_unit_number(
                record["default_suppression_threshold"],
                f"suppression threshold for {name}",
            ),
        )
        known_aliases.update(aliases)
    if not entity_types:
        raise OntologyValidationError("entity_types must not be empty")

    relations: dict[str, RelationDefinition] = {}
    for record in _records(root.get("relations"), "relations"):
        _require_fields(record, _RELATION_FIELDS, "relation")
        missing = _RELATION_FIELDS.difference(record)
        if missing:
            raise OntologyValidationError(
                f"relation is missing fields: {sorted(missing)}"
            )
        name = _nonempty_string(record["name"], "relation name")
        if name in relations:
            raise OntologyValidationError(f"duplicate relation name: {name}")
        direction = _nonempty_string(record["direction"], f"direction for {name}")
        if direction not in _DIRECTIONS:
            raise OntologyValidationError(f"invalid direction for {name}")
        heads = _names(record["allowed_head_types"], f"head types for {name}")
        tails = _names(record["allowed_tail_types"], f"tail types for {name}")
        unknown = set(heads).union(tails).difference(entity_types)
        if unknown:
            raise OntologyValidationError(
                f"unknown endpoint types for {name}: {sorted(unknown)}"
            )
        relations[name] = RelationDefinition(
            name=name,
            description=_nonempty_string(
                record["description"], f"description for {name}"
            ),
            direction=direction,
            allowed_head_types=heads,
            allowed_tail_types=tails,
        )
    if not relations:
        raise OntologyValidationError("relations must not be empty")
    return _build_definition(version, entity_types, relations, raw_yaml)


def load_ontology_extension(path: str | Path) -> OntologyExtensionDefinition:
    """Load a partial, versioned change set without applying it."""
    document, raw_yaml = _read_yaml(path)
    root = _mapping(document, "ontology extension")
    _require_fields(
        root, frozenset({"version", "entity_types", "relations"}), "ontology extension"
    )
    version = _semantic_version(root.get("version"))
    entities: dict[str, EntityTypeExtension] = {}
    for record in _records(root.get("entity_types", []), "entity_types"):
        _require_fields(record, _ENTITY_FIELDS, "entity type extension")
        name = _nonempty_string(record.get("name"), "entity type extension name")
        if name in entities:
            raise OntologyValidationError(f"duplicate entity extension: {name}")
        entities[name] = EntityTypeExtension(
            name=name,
            description=(
                None
                if "description" not in record
                else _nonempty_string(record["description"], f"description for {name}")
            ),
            aliases=(
                None
                if "aliases" not in record
                else _aliases(record["aliases"], f"aliases for {name}")
            ),
            default_retrieval_weight=(
                None
                if "default_retrieval_weight" not in record
                else _unit_number(
                    record["default_retrieval_weight"], f"retrieval weight for {name}"
                )
            ),
            default_suppression_policy=(
                None
                if "default_suppression_policy" not in record
                else _nonempty_string(
                    record["default_suppression_policy"],
                    f"suppression policy for {name}",
                )
            ),
            default_suppression_threshold=(
                None
                if "default_suppression_threshold" not in record
                else _unit_number(
                    record["default_suppression_threshold"],
                    f"suppression threshold for {name}",
                )
            ),
        )
    relations: dict[str, RelationExtension] = {}
    for record in _records(root.get("relations", []), "relations"):
        _require_fields(record, _RELATION_FIELDS, "relation extension")
        name = _nonempty_string(record.get("name"), "relation extension name")
        if name in relations:
            raise OntologyValidationError(f"duplicate relation extension: {name}")
        direction = (
            None
            if "direction" not in record
            else _nonempty_string(record["direction"], f"direction for {name}")
        )
        if direction is not None and direction not in _DIRECTIONS:
            raise OntologyValidationError(f"invalid direction for {name}")
        relations[name] = RelationExtension(
            name=name,
            description=(
                None
                if "description" not in record
                else _nonempty_string(record["description"], f"description for {name}")
            ),
            direction=direction,
            allowed_head_types=(
                None
                if "allowed_head_types" not in record
                else _names(record["allowed_head_types"], f"head types for {name}")
            ),
            allowed_tail_types=(
                None
                if "allowed_tail_types" not in record
                else _names(record["allowed_tail_types"], f"tail types for {name}")
            ),
        )
    content = {
        "version": version,
        "entity_types": [
            {key: value for key, value in asdict(entity).items() if value is not None}
            for entity in (entities[name] for name in sorted(entities))
        ],
        "relations": [
            {key: value for key, value in asdict(relation).items() if value is not None}
            for relation in (relations[name] for name in sorted(relations))
        ],
    }
    return OntologyExtensionDefinition(
        version=version,
        entity_types=MappingProxyType(dict(sorted(entities.items()))),
        relations=MappingProxyType(dict(sorted(relations.items()))),
        checksum=_checksum(content),
        canonical_yaml=_canonical_yaml(content),
        raw_yaml=raw_yaml,
    )


def merge_ontology_extension(
    base: OntologyDefinition, delta: OntologyExtensionDefinition
) -> OntologyDefinition:
    """Return a new immutable ontology after a core-preserving extension merge."""
    base_version = _semantic_version(base.version)
    delta_version = _semantic_version(delta.version)
    if _compare_semver_precedence(delta_version, base_version) <= 0:
        raise OntologyValidationError(
            "extension version must be strictly newer than the base version"
        )
    entity_types = dict(base.entity_types)
    relations = dict(base.relations)
    for name, extension in delta.entity_types.items():
        previous = entity_types.get(name)
        if previous is None:
            if None in (
                extension.description,
                extension.aliases,
                extension.default_retrieval_weight,
                extension.default_suppression_policy,
                extension.default_suppression_threshold,
            ):
                raise OntologyValidationError(
                    f"new entity type {name} requires a complete definition"
                )
            entity_types[name] = EntityTypeDefinition(
                name=name,
                description=extension.description,
                aliases=extension.aliases,
                default_retrieval_weight=extension.default_retrieval_weight,
                default_suppression_policy=extension.default_suppression_policy,
                default_suppression_threshold=extension.default_suppression_threshold,
            )
            continue
        if (
            extension.description is not None
            and extension.description != previous.description
        ):
            raise OntologyValidationError(
                f"extension redefines core entity type {name}"
            )
        entity_types[name] = EntityTypeDefinition(
            name=name,
            description=previous.description,
            aliases=tuple(sorted(set(previous.aliases).union(extension.aliases or ()))),
            default_retrieval_weight=(
                extension.default_retrieval_weight
                if extension.default_retrieval_weight is not None
                else previous.default_retrieval_weight
            ),
            default_suppression_policy=(
                extension.default_suppression_policy
                if extension.default_suppression_policy is not None
                else previous.default_suppression_policy
            ),
            default_suppression_threshold=(
                extension.default_suppression_threshold
                if extension.default_suppression_threshold is not None
                else previous.default_suppression_threshold
            ),
        )
    all_names = set(entity_types)
    all_aliases = [
        alias for entity in entity_types.values() for alias in entity.aliases
    ]
    if len(all_aliases) != len(set(all_aliases)) or all_names.intersection(all_aliases):
        raise OntologyValidationError(
            "extension introduces duplicate entity names or aliases"
        )

    for name, extension in delta.relations.items():
        previous = relations.get(name)
        if previous is None:
            if None in (
                extension.description,
                extension.direction,
                extension.allowed_head_types,
                extension.allowed_tail_types,
            ):
                raise OntologyValidationError(
                    f"new relation {name} requires a complete definition"
                )
            relation = RelationDefinition(
                name,
                extension.description,
                extension.direction,
                extension.allowed_head_types,
                extension.allowed_tail_types,
            )
        else:
            if (
                (
                    extension.description is not None
                    and extension.description != previous.description
                )
                or (
                    extension.direction is not None
                    and extension.direction != previous.direction
                )
                or (
                    extension.allowed_head_types is not None
                    and extension.allowed_head_types != previous.allowed_head_types
                )
                or (
                    extension.allowed_tail_types is not None
                    and extension.allowed_tail_types != previous.allowed_tail_types
                )
            ):
                raise OntologyValidationError(
                    f"extension redefines core relation {name}"
                )
            relation = previous
        unknown = (
            set(relation.allowed_head_types)
            .union(relation.allowed_tail_types)
            .difference(all_names)
        )
        if unknown:
            raise OntologyValidationError(
                f"unknown endpoint types for {name}: {sorted(unknown)}"
            )
        relations[name] = relation
    return _build_definition(
        delta_version,
        entity_types,
        relations,
        raw_yaml=None,
        provenance={
            "base_checksum": base.checksum,
            "base_version": base_version,
            "delta_checksum": delta.checksum,
            "delta_version": delta_version,
        },
    )


def _lock_graph_ontology_activation(cursor: Any) -> None:
    """Serialize graph ontology activation, including an initially empty table."""
    cursor.execute(
        "SELECT pg_advisory_xact_lock(%s)", [_GRAPH_ONTOLOGY_ACTIVATION_LOCK]
    )


def activate_ontology(definition: OntologyDefinition):
    """Persist and activate a definition atomically, without provider calls."""
    from django.db import IntegrityError, connection, transaction
    from django.utils import timezone

    from apps.knowledge_graph.models import OntologyVersion

    yaml_text = definition.raw_yaml or definition.canonical_yaml
    if not yaml_text:
        raise OntologyValidationError("ontology activation requires nonempty YAML")
    metadata = {
        "yaml": yaml_text,
        "canonical_yaml": definition.canonical_yaml,
        "provenance": dict(definition.provenance),
        "checksum_algorithm": "sha256-canonical-json-v1",
    }
    with transaction.atomic():
        with connection.cursor() as cursor:
            _lock_graph_ontology_activation(cursor)
        conflict_message = (
            f"version {definition.version} is already persisted with a different "
            "checksum"
        )
        record = (
            OntologyVersion.objects.select_for_update()
            .filter(kind=OntologyVersion.Kind.GRAPH, version=definition.version)
            .first()
        )
        if record is not None and record.checksum != definition.checksum:
            raise OntologyValidationError(conflict_message)
        if record is None:
            try:
                with transaction.atomic():
                    record = OntologyVersion.objects.create(
                        kind=OntologyVersion.Kind.GRAPH,
                        version=definition.version,
                        checksum=definition.checksum,
                        metadata=metadata,
                        status=OntologyVersion.Status.DRAFT,
                    )
            except IntegrityError:
                record = OntologyVersion.objects.select_for_update().get(
                    kind=OntologyVersion.Kind.GRAPH, version=definition.version
                )
                if record.checksum != definition.checksum:
                    raise OntologyValidationError(conflict_message)
        OntologyVersion.objects.filter(
            kind=OntologyVersion.Kind.GRAPH, status=OntologyVersion.Status.ACTIVE
        ).exclude(pk=record.pk).update(status=OntologyVersion.Status.SUPERSEDED)
        if record.status != OntologyVersion.Status.ACTIVE:
            record.status = OntologyVersion.Status.ACTIVE
            record.activated_at = timezone.now()
            record.save(update_fields=["status", "activated_at"])
        return record
