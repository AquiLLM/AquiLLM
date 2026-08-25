from __future__ import annotations

import json
import uuid

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


def body_positive_int(body: dict, field: str, error_code: str) -> int:
    value = body.get(field)
    if type(value) is not int or value <= 0:
        raise SchemaOperationError(error_code)
    return value


def body_nonempty_string(body: dict, field: str, error_code: str) -> str:
    value = body.get(field)
    if type(value) is not str or not value:
        raise SchemaOperationError(error_code)
    return value


def body_uuid_string(body: dict, field: str, error_code: str) -> str:
    value = body_nonempty_string(body, field, error_code)
    try:
        uuid.UUID(value)
    except ValueError as exc:
        raise SchemaOperationError(error_code) from exc
    return value


def matching_body_revision(request, body: dict, field: str) -> int:
    revision = body_positive_int(body, field, "invalid_revision")
    if parse_revision(request) != revision:
        raise SchemaOperationError("revision_mismatch")
    return revision


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
    try:
        value = json.loads(request.body)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise SchemaOperationError("invalid_json") from exc
    if type(value) is not dict:
        raise SchemaOperationError("invalid_body")
    return value


__all__ = [
    "body_nonempty_string",
    "body_positive_int",
    "body_uuid_string",
    "conflict_response",
    "error_response",
    "load_body",
    "matching_body_revision",
    "parse_revision",
    "permission_level",
    "permissions_snapshot",
    "require_edit",
    "require_manage",
    "require_view",
    "workspace_envelope",
]
