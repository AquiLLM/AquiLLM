from __future__ import annotations

import json

from django.http import JsonResponse

from apps.collections.models import Collection
from apps.collections.services.schema import (
    SchemaOperationError,
    SchemaRevisionConflict,
    permission_level,
    workspace_envelope,
)


def permissions_snapshot(level: str) -> dict:
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


def require_view(collection: Collection, user) -> JsonResponse | None:
    if not collection.user_can_view(user):
        return JsonResponse({"error": "forbidden"}, status=403)
    return None


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


def conflict_response(conflict: SchemaRevisionConflict) -> JsonResponse:
    return JsonResponse(
        {
            "attempted_revision": conflict.attempted,
            "current_revision": conflict.draft.revision,
            "draft_id": str(conflict.draft.pk),
            "definitions": conflict.definitions,
        },
        status=409,
    )


def error_response(error: SchemaOperationError) -> JsonResponse:
    return JsonResponse({"error": error.code}, status=error.status)


def load_body(request) -> dict:
    if not request.body:
        return {}
    value = json.loads(request.body)
    if type(value) is not dict:
        raise SchemaOperationError("invalid_body")
    return value


__all__ = [
    "conflict_response",
    "error_response",
    "load_body",
    "parse_revision",
    "permission_level",
    "permissions_snapshot",
    "require_edit",
    "require_manage",
    "require_view",
    "workspace_envelope",
]
