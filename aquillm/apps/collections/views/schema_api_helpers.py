"""Shared helpers for collection schema API stub views."""
from __future__ import annotations

import json
from copy import deepcopy

from django.http import JsonResponse

from apps.collections.models import Collection

from .schema_api_fixtures import CONSTRAINTS, PUBLISHED_ENTITY, PUBLISHED_RELATION


def permission_level(collection: Collection, user) -> str:
    if collection.user_can_manage(user):
        return "MANAGE"
    if collection.user_can_edit(user):
        return "EDIT"
    return "VIEW"


def permissions_snapshot(level: str) -> dict:
    return {
        "level": level,
        "can_create_draft": level in {"EDIT", "MANAGE"},
        "can_edit_definitions": level in {"EDIT", "MANAGE"},
        "can_validate": level in {"EDIT", "MANAGE"},
        "can_publish": level == "MANAGE",
        "can_discard_draft": level == "MANAGE",
        "can_restore": level == "MANAGE",
        "can_view_history": True,
    }


def draft_snapshot(level: str, revision: int = 2) -> dict | None:
    if level == "VIEW":
        return None
    entity = deepcopy(PUBLISHED_ENTITY)
    entity["change_state"] = "changed"
    entity["values"] = {**entity["values"], "description": "Updated person description"}
    return {
        "draft_id": f"draft-{level.lower()}-1",
        "revision": revision,
        "base_published_checksum": "pub-edit-checksum",
        "last_editor": "editor@example.test",
        "updated_at": "2026-08-21T10:00:00Z",
        "entities": [entity],
        "relations": [deepcopy(PUBLISHED_RELATION)],
    }


def workspace_envelope(collection: Collection, user, revision: int = 2) -> dict:
    level = permission_level(collection, user)
    return {
        "collection_id": str(collection.id),
        "permissions": permissions_snapshot(level),
        "published": {
            "version": 4,
            "checksum": "pub-edit-checksum",
            "entities": [deepcopy(PUBLISHED_ENTITY)],
            "relations": [deepcopy(PUBLISHED_RELATION)],
        },
        "draft": draft_snapshot(level, revision),
        "constraints": deepcopy(CONSTRAINTS),
    }


def require_edit(collection: Collection, user) -> JsonResponse | None:
    if not collection.user_can_edit(user):
        return JsonResponse({"error": "forbidden"}, status=403)
    return None


def require_manage(collection: Collection, user) -> JsonResponse | None:
    if not collection.user_can_manage(user):
        return JsonResponse({"error": "forbidden"}, status=403)
    return None


def parse_revision(request) -> int | None:
    header = request.headers.get("If-Match", "").strip()
    if not header:
        return None
    try:
        return int(header)
    except ValueError:
        return None


def conflict_response(attempted: int, current: int = 6) -> JsonResponse:
    return JsonResponse(
        {
            "attempted_revision": attempted,
            "current_revision": current,
            "draft_id": "draft-manage-1",
            "definitions": [
                {
                    "kind": "entity",
                    "key": "person",
                    "fields": [
                        {
                            "field": "description",
                            "server_value": "Server accepted description",
                            "attempted_value": "Local unsaved description",
                        }
                    ],
                }
            ],
        },
        status=409,
    )


def load_body(request) -> dict:
    if not request.body:
        return {}
    return json.loads(request.body)
