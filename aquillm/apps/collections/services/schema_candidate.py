"""Normalize model proposals into validated editor definitions."""

from __future__ import annotations

import re

import yaml

from .schema_generation import (
    _MAX_ENTITY_TYPES,
    _MAX_RELATION_TYPES,
    _MIN_ENTITY_TYPES,
    _MIN_RELATION_TYPES,
    InvalidSchemaCandidate,
)

_MAX_ALIASES_PER_ENTITY = 16
_MAX_ALIAS_CHARACTERS = 128
_MAX_ALIAS_TOTAL_CHARACTERS = 1_024


def _snake_case(value: object, field: str) -> str:
    from lib.knowledge_graph.type_names import validate_type_name

    if not isinstance(value, str) or not value.strip():
        raise InvalidSchemaCandidate(f"{field} must be a nonempty string")
    normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    try:
        validate_type_name(normalized, field)
    except ValueError as exc:
        raise InvalidSchemaCandidate(str(exc)) from exc
    return normalized


def _description(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > 512:
        raise InvalidSchemaCandidate(
            f"{field} must be a nonempty string up to 512 characters"
        )
    return value.strip()


def _aliases(value: object) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(alias, str) or not alias.strip() for alias in value
    ):
        raise InvalidSchemaCandidate("aliases must be a list of nonempty strings")
    aliases = sorted(set(alias.strip() for alias in value))
    if len(aliases) > _MAX_ALIASES_PER_ENTITY:
        raise InvalidSchemaCandidate("aliases exceed the per-entity count limit")
    if any(len(alias) > _MAX_ALIAS_CHARACTERS for alias in aliases):
        raise InvalidSchemaCandidate("aliases exceed the per-alias character limit")
    if sum(len(alias) for alias in aliases) > _MAX_ALIAS_TOTAL_CHARACTERS:
        raise InvalidSchemaCandidate("aliases exceed the per-entity character limit")
    return aliases


def _candidate_ontology_yaml(definitions: dict) -> str:
    entity_types = [
        {
            "name": item["key"],
            "description": item["values"]["description"],
            "aliases": item["values"]["aliases"],
            "default_retrieval_weight": item["values"]["default_retrieval_weight"],
            "default_suppression_policy": item["values"]["default_suppression_policy"],
            "default_suppression_threshold": item["values"][
                "default_suppression_threshold"
            ],
        }
        for item in definitions["entities"]
    ]
    relations = [
        {
            "name": item["key"],
            "description": item["values"]["description"],
            "direction": item["values"]["direction"],
            "allowed_head_types": item["values"]["allowed_head_types"],
            "allowed_tail_types": item["values"]["allowed_tail_types"],
        }
        for item in definitions["relations"]
    ]
    return yaml.safe_dump(
        {"version": "1.0.0", "entity_types": entity_types, "relations": relations},
        sort_keys=True,
    )


def normalize_schema_candidate(candidate: object) -> dict:
    """Convert strict model JSON into the canonical editor definition shape."""

    if not isinstance(candidate, dict):
        raise InvalidSchemaCandidate("candidate must be a JSON object")
    entity_records, relation_records = (
        candidate.get("entities"),
        candidate.get("relations"),
    )
    if not isinstance(entity_records, list) or not (
        _MIN_ENTITY_TYPES <= len(entity_records) <= _MAX_ENTITY_TYPES
    ):
        raise InvalidSchemaCandidate("candidate must contain 2-24 entities")
    if not isinstance(relation_records, list) or not (
        _MIN_RELATION_TYPES <= len(relation_records) <= _MAX_RELATION_TYPES
    ):
        raise InvalidSchemaCandidate("candidate must contain 1-32 relations")
    entities, entity_keys = [], set()
    for record in entity_records:
        if not isinstance(record, dict):
            raise InvalidSchemaCandidate("entity definitions must be objects")
        key = _snake_case(record.get("name"), "entity name")
        if key in entity_keys:
            raise InvalidSchemaCandidate("duplicate entity name")
        entity_keys.add(key)
        entities.append(
            {
                "key": key,
                "origin": "generated",
                "change_state": "added",
                "capabilities": {
                    "editable_fields": [
                        "description",
                        "aliases",
                        "default_retrieval_weight",
                        "default_suppression_policy",
                        "default_suppression_threshold",
                    ],
                    "removable": True,
                    "renameable": False,
                },
                "values": {
                    "name": key,
                    "description": _description(
                        record.get("description"), "entity description"
                    ),
                    "aliases": _aliases(record.get("aliases")),
                    "default_retrieval_weight": 0.5,
                    "default_suppression_policy": "none",
                    "default_suppression_threshold": 0.0,
                },
            }
        )
    relations, relation_keys = [], set()
    for record in relation_records:
        if not isinstance(record, dict):
            raise InvalidSchemaCandidate("relation definitions must be objects")
        key = _snake_case(record.get("name"), "relation name")
        if key in relation_keys:
            raise InvalidSchemaCandidate("duplicate relation name")
        relation_keys.add(key)
        direction = record.get("direction")
        if direction not in {"directed", "undirected"}:
            raise InvalidSchemaCandidate(
                "relation direction must be directed or undirected"
            )
        heads = [
            _snake_case(value, "relation head endpoint")
            for value in record.get("allowed_head_types", [])
        ]
        tails = [
            _snake_case(value, "relation tail endpoint")
            for value in record.get("allowed_tail_types", [])
        ]
        if not heads or not tails or set(heads).union(tails).difference(entity_keys):
            raise InvalidSchemaCandidate("relation has an unknown endpoint type")
        relations.append(
            {
                "key": key,
                "origin": "generated",
                "change_state": "added",
                "capabilities": {
                    "editable_fields": [
                        "description",
                        "direction",
                        "allowed_head_types",
                        "allowed_tail_types",
                    ],
                    "removable": True,
                    "renameable": False,
                },
                "values": {
                    "name": key,
                    "description": _description(
                        record.get("description"), "relation description"
                    ),
                    "direction": direction,
                    "allowed_head_types": sorted(set(heads)),
                    "allowed_tail_types": sorted(set(tails)),
                },
            }
        )
    definitions = {
        "entities": sorted(entities, key=lambda item: item["key"]),
        "relations": sorted(relations, key=lambda item: item["key"]),
    }
    try:
        from apps.knowledge_graph.services.ontology import load_ontology_yaml

        load_ontology_yaml(_candidate_ontology_yaml(definitions))
    except Exception as exc:
        raise InvalidSchemaCandidate("candidate fails ontology validation") from exc
    return definitions
