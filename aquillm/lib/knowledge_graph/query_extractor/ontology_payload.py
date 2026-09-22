"""Bounded, provider-neutral ontology transport with canonical identity checks."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from types import MappingProxyType

from lib.knowledge_graph.type_names import validate_type_name

MAX_ONTOLOGY_DEFINITION_BYTES = 65_536
MAX_ENTITY_TYPES = 64
MAX_RELATION_TYPES = 128
MAX_ALIASES_PER_TYPE = 32
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
_SEMVER = re.compile(
    r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
    r"(?:-(?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*)"
    r"(?:\.(?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*))*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
)


def _encoded(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _text(value: object, maximum: int) -> str:
    if type(value) is not str or not value.strip() or len(value) > maximum:
        raise ValueError(
            f"ontology text must be nonempty and at most {maximum} characters"
        )
    value.encode("utf-8")
    return value.strip()


def _unit(value: object) -> float:
    if type(value) not in (int, float) or not isfinite(value) or not 0 <= value <= 1:
        raise ValueError("ontology weight must be a finite unit number")
    return float(value)


def _records(value: object, maximum: int) -> list[dict[str, object]]:
    if type(value) is not list or not 1 <= len(value) <= maximum:
        raise ValueError(f"ontology definition count must be 1..{maximum}")
    if any(type(row) is not dict for row in value):
        raise ValueError("ontology definitions must be exact objects")
    return value


def _strings(value: object, maximum: int, *, nonempty: bool = False) -> list[str]:
    if type(value) is not list or len(value) > maximum or (nonempty and not value):
        raise ValueError(
            f"ontology string list must contain {1 if nonempty else 0}..{maximum} items"
        )
    strings = [_text(item, 128) for item in value]
    if len(set(strings)) != len(strings):
        raise ValueError("ontology string list contains duplicates")
    return sorted(strings)


def _freeze(row: dict[str, object]) -> Mapping[str, object]:
    return MappingProxyType(
        {
            key: tuple(value) if type(value) is list else value
            for key, value in row.items()
        }
    )


@dataclass(frozen=True, slots=True)
class QueryOntologyDefinition:
    version: str
    entity_types: Mapping[str, Mapping[str, object]]
    relations: Mapping[str, Mapping[str, object]]
    checksum: str
    canonical_json: bytes


def load_ontology_definition(
    document: object, *, expected_checksum: str, require_canonical: bool = True
) -> QueryOntologyDefinition:
    """Validate transport before inference and match the persisted semantic hash."""
    if type(document) is not dict or set(document) != {
        "version",
        "entity_types",
        "relations",
    }:
        raise ValueError("ontology has an invalid field set")
    if len(_encoded(document)) > MAX_ONTOLOGY_DEFINITION_BYTES:
        raise ValueError(
            "ontology definition exceeds its byte cap of "
            f"{MAX_ONTOLOGY_DEFINITION_BYTES}"
        )
    version = _text(document["version"], 128)
    if _SEMVER.fullmatch(version) is None:
        raise ValueError("ontology version must be semantic")
    entities = []
    names: set[str] = set()
    aliases_seen: set[str] = set()
    for row in _records(document["entity_types"], MAX_ENTITY_TYPES):
        if not _ENTITY_FIELDS <= set(row) <= _ENTITY_FIELDS | {"extension_enabled"}:
            raise ValueError("entity type has an invalid field set")
        name = validate_type_name(row["name"])
        aliases = _strings(row["aliases"], MAX_ALIASES_PER_TYPE)
        if (
            name in names | aliases_seen
            or name in aliases
            or set(aliases) & (names | aliases_seen)
        ):
            raise ValueError("duplicate entity name or alias")
        extension = row.get("extension_enabled", False)
        if type(extension) is not bool:
            raise ValueError("extension_enabled must be an exact boolean")
        entities.append(
            {
                "name": name,
                "description": _text(row["description"], 512),
                "aliases": aliases,
                "default_retrieval_weight": _unit(row["default_retrieval_weight"]),
                "default_suppression_policy": _text(
                    row["default_suppression_policy"], 128
                ),
                "default_suppression_threshold": _unit(
                    row["default_suppression_threshold"]
                ),
                "extension_enabled": extension,
            }
        )
        names.add(name)
        aliases_seen.update(aliases)
    relations = []
    relation_names: set[str] = set()
    for row in _records(document["relations"], MAX_RELATION_TYPES):
        if set(row) != _RELATION_FIELDS:
            raise ValueError("relation has an invalid field set")
        name = validate_type_name(row["name"])
        if name in relation_names:
            raise ValueError("duplicate relation name")
        direction = _text(row["direction"], 16)
        if direction not in {"directed", "undirected"}:
            raise ValueError("relation direction is invalid")
        heads = _strings(row["allowed_head_types"], MAX_ENTITY_TYPES, nonempty=True)
        tails = _strings(row["allowed_tail_types"], MAX_ENTITY_TYPES, nonempty=True)
        if not set(heads + tails) <= names:
            raise ValueError("relation has a foreign endpoint type")
        relations.append(
            {
                "name": name,
                "description": _text(row["description"], 512),
                "direction": direction,
                "allowed_head_types": heads,
                "allowed_tail_types": tails,
            }
        )
        relation_names.add(name)
    entities.sort(key=lambda row: row["name"])
    relations.sort(key=lambda row: row["name"])
    canonical = _encoded(
        {"version": version, "entity_types": entities, "relations": relations}
    )
    checksum = sha256(canonical).hexdigest()
    if type(expected_checksum) is not str or checksum != expected_checksum:
        raise ValueError("ontology definition checksum mismatch")
    if require_canonical and _encoded(document) != canonical:
        raise ValueError("ontology definition must use its canonical semantic form")
    return QueryOntologyDefinition(
        version=version,
        entity_types=MappingProxyType({row["name"]: _freeze(row) for row in entities}),
        relations=MappingProxyType({row["name"]: _freeze(row) for row in relations}),
        checksum=checksum,
        canonical_json=canonical,
    )


def ontology_definition_payload(ontology: object) -> dict[str, object] | None:
    """Serialize full definitions; keep checksum-only legacy clients supported."""
    if not hasattr(ontology, "version") or not hasattr(ontology, "relations"):
        return None

    def records(definitions, fields, *, entity=False):
        if not isinstance(definitions, Mapping):
            raise ValueError("ontology definitions must be a mapping")
        result = []
        for key, definition in definitions.items():

            def field(name, default=None):
                if isinstance(definition, Mapping):
                    return definition.get(name, default)
                return getattr(definition, name, default)

            row = {name: field(name) for name in fields}
            if row["name"] != key:
                raise ValueError(
                    "ontology definition name differs from its mapping key"
                )
            for name in ("aliases", "allowed_head_types", "allowed_tail_types"):
                if name in row and type(row[name]) in (list, tuple):
                    row[name] = list(row[name])
            if entity:
                row["extension_enabled"] = field("extension_enabled", False)
            result.append(row)
        return result

    document = {
        "version": ontology.version,
        "entity_types": records(ontology.entity_types, _ENTITY_FIELDS, entity=True),
        "relations": records(ontology.relations, _RELATION_FIELDS),
    }
    validated = load_ontology_definition(
        document, expected_checksum=ontology.checksum, require_canonical=False
    )
    return json.loads(validated.canonical_json)
