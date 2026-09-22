# ruff: noqa: E402, I001 - facade imports follow the shared definitions they require
from __future__ import annotations

import json
from copy import deepcopy
from hashlib import sha256
from typing import Any

import structlog
from django.db import transaction  # noqa: F401 - exposed for transaction patching

from apps.collections.models import (
    Collection,
    CollectionSchemaDraft as CollectionSchemaDraft,
    CollectionSchemaVersion,
)
from lib.knowledge_graph.type_names import (
    PROVIDER_RESERVED_TYPE_NAMES,
    TYPE_NAME_MAX_LENGTH,
    TYPE_NAME_PATTERN,
)

logger = structlog.stdlib.get_logger(__name__)

_NAME_CONSTRAINT = {
    "required": True,
    "max_length": TYPE_NAME_MAX_LENGTH,
    "pattern": TYPE_NAME_PATTERN,
    "disallowed_values": sorted(PROVIDER_RESERVED_TYPE_NAMES),
}
CONSTRAINTS = {
    "entity_fields": {
        "name": _NAME_CONSTRAINT,
        "description": {"max_length": 512},
        "default_retrieval_weight": {"min": 0, "max": 1},
        "default_suppression_threshold": {"min": 0, "max": 1},
    },
    "relation_fields": {
        "name": _NAME_CONSTRAINT,
        "description": {"max_length": 512},
        "direction": {"allowed_values": ["directed", "undirected"]},
    },
}


class SchemaGenerationDraftConflict(RuntimeError):
    """The shared draft no longer matches a generation run's exact base."""


class SchemaRevisionConflict(RuntimeError):
    def __init__(self, attempted: int | None, draft, definitions=None):
        self.attempted = attempted
        self.draft = draft
        self.definitions = definitions or []
        super().__init__("schema draft revision conflict")


class SchemaOperationError(ValueError):
    def __init__(self, code: str, *, status: int = 400):
        self.code = code
        self.status = status
        super().__init__(code)


def canonicalize_definitions(definitions: dict[str, Any]) -> dict[str, list[dict]]:
    if type(definitions) is not dict:
        raise ValueError("schema definitions must be an object")
    entities = definitions.get("entities", [])
    relations = definitions.get("relations", [])
    if type(entities) is not list or type(relations) is not list:
        raise ValueError("schema entities and relations must be arrays")
    for label, rows in (("entity", entities), ("relation", relations)):
        if any(type(row) is not dict for row in rows):
            raise ValueError(f"schema {label} definitions must be objects")
        keys = [row.get("key") for row in rows]
        if any(type(key) is not str or not key for key in keys):
            raise ValueError(f"schema {label} keys must be nonempty strings")
        if len(keys) != len(set(keys)):
            raise ValueError(f"schema {label} keys must be unique")
    result = {
        "entities": sorted(deepcopy(entities), key=lambda row: row["key"]),
        "relations": sorted(deepcopy(relations), key=lambda row: row["key"]),
    }
    json.dumps(result, allow_nan=False)
    return result


def definitions_checksum(definitions: dict[str, Any]) -> str:
    canonical = canonicalize_definitions(definitions)
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _permission_level(collection: Collection, user) -> str:
    if collection.user_can_manage(user):
        return "MANAGE"
    if collection.user_can_edit(user):
        return "EDIT"
    return "VIEW"


def permission_level(collection: Collection, user) -> str:
    return _permission_level(collection, user)


def _permissions(level: str) -> dict[str, bool | str]:
    editable = level in {"EDIT", "MANAGE"}
    manageable = level == "MANAGE"
    return {
        "level": level,
        "can_create_draft": editable,
        "can_edit_definitions": editable,
        "can_validate": editable,
        "can_publish": manageable,
        "can_discard_draft": manageable,
        "can_restore": manageable,
        "can_view_history": True,
    }


def _current_version(collection: Collection) -> CollectionSchemaVersion | None:
    version_id = collection.current_schema_version_id
    if version_id is None:
        return None
    return CollectionSchemaVersion.objects.filter(
        collection=collection,
        pk=version_id,
    ).first()


def workspace_envelope(collection: Collection, user) -> dict[str, Any]:
    level = _permission_level(collection, user)
    published = _current_version(collection)
    published_definitions = (
        _published_definitions(published.definitions)
        if published is not None
        else {"entities": [], "relations": []}
    )
    draft_payload = None
    if level != "VIEW":
        draft = getattr(collection, "schema_draft", None)
        if draft is not None:
            definitions = _editor_definitions(draft.definitions)
            draft_payload = {
                "draft_id": str(draft.pk),
                "revision": draft.revision,
                "base_published_checksum": (
                    draft.base_version.checksum
                    if draft.base_version is not None
                    else ""
                ),
                "last_editor": draft.last_editor.get_username(),
                "updated_at": draft.updated_at.isoformat(),
                **definitions,
            }
    return {
        "collection_id": str(collection.pk),
        "permissions": _permissions(level),
        "published": {
            "version": published.version if published is not None else 0,
            "checksum": published.checksum if published is not None else "",
            **published_definitions,
        },
        "draft": draft_payload,
        "constraints": deepcopy(CONSTRAINTS),
    }


def _capabilities(kind: str) -> dict[str, Any]:
    fields = (
        [
            "description",
            "aliases",
            "default_retrieval_weight",
            "default_suppression_policy",
            "default_suppression_threshold",
        ]
        if kind == "entity"
        else [
            "description",
            "direction",
            "allowed_head_types",
            "allowed_tail_types",
        ]
    )
    return {"editable_fields": fields, "removable": True, "renameable": False}


def _editor_definitions(definitions: dict[str, Any]) -> dict[str, list[dict]]:
    """Derive truthful editor capabilities without mutating immutable snapshots."""

    canonical = canonicalize_definitions(definitions)
    for kind, rows in (
        ("entity", canonical["entities"]),
        ("relation", canonical["relations"]),
    ):
        for row in rows:
            row["capabilities"] = _capabilities(kind)
    return canonical


def _published_definitions(definitions: dict[str, Any]) -> dict[str, list[dict]]:
    canonical = _editor_definitions(definitions)
    for kind in ("entities", "relations"):
        for row in canonical[kind]:
            row["change_state"] = "unchanged"
    return canonical


def _candidate_definitions(definitions: dict[str, Any]) -> dict[str, list[dict]]:
    return _published_definitions(definitions)


def _next_version(collection: Collection) -> int:
    current = (
        CollectionSchemaVersion.objects.filter(collection=collection)
        .order_by("-version")
        .values_list("version", flat=True)
        .first()
    )
    return (current or 0) + 1


from .schema_drafts import (
    _locked_draft as _locked_draft,
    create_draft as create_draft,
    discard_draft as discard_draft,
    mutate_definition as mutate_definition,
    write_generated_draft as write_generated_draft,
)
from .schema_publication import (
    _diff_summary as _diff_summary,
    diff_definitions as diff_definitions,
    publish_draft as publish_draft,
    validate_draft as validate_draft,
)
from .schema_history import (
    draft_diff as draft_diff,
    history_page as history_page,
    replace_with_version as replace_with_version,
    restore_version as restore_version,
    version_diff as version_diff,
)
