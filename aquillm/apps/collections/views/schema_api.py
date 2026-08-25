from __future__ import annotations

from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from django.views.decorators.http import require_http_methods

from apps.collections.models import Collection
from apps.collections.services import schema as schema_service

from .schema_api_helpers import (
    conflict_response,
    error_response,
    load_body,
    matching_body_revision,
    parse_revision,
    require_edit,
    require_manage,
    require_view,
    workspace_envelope,
)


def _collection(col_id: int) -> Collection:
    return get_object_or_404(Collection, id=col_id)


def _failure(exc: Exception):
    if isinstance(exc, schema_service.SchemaRevisionConflict):
        return conflict_response(exc)
    if isinstance(exc, schema_service.SchemaOperationError):
        return error_response(exc)
    raise exc


@login_required
@require_http_methods(["GET"])
def schema_workspace(request, col_id: int):
    collection = _collection(col_id)
    if denied := require_view(collection, request.user):
        return denied
    return JsonResponse(workspace_envelope(collection, request.user))


@login_required
@require_http_methods(["POST"])
def schema_create_draft(request, col_id: int):
    collection = _collection(col_id)
    if denied := require_edit(collection, request.user):
        return denied
    schema_service.create_draft(collection, request.user)
    return JsonResponse(workspace_envelope(collection, request.user))


def _mutate(request, col_id: int, kind: str, key: str):
    collection = _collection(col_id)
    if denied := require_edit(collection, request.user):
        return denied
    try:
        values = None
        if request.method == "PUT":
            body = load_body(request)
            values = body.get("values")
            if "values" not in body or type(values) is not dict:
                raise schema_service.SchemaOperationError("invalid_definition")
        schema_service.mutate_definition(
            collection,
            request.user,
            kind,
            key,
            parse_revision(request),
            values,
        )
    except (
        schema_service.SchemaRevisionConflict,
        schema_service.SchemaOperationError,
    ) as exc:
        return _failure(exc)
    return JsonResponse(workspace_envelope(collection, request.user))


@login_required
@require_http_methods(["PUT", "DELETE"])
def schema_entity(request, col_id: int, entity_key: str):
    return _mutate(request, col_id, "entity", entity_key)


@login_required
@require_http_methods(["PUT", "DELETE"])
def schema_relation(request, col_id: int, relation_key: str):
    return _mutate(request, col_id, "relation", relation_key)


@login_required
@require_http_methods(["POST"])
def schema_validate(request, col_id: int):
    collection = _collection(col_id)
    if denied := require_edit(collection, request.user):
        return denied
    try:
        body = load_body(request)
        result = schema_service.validate_draft(
            collection, body.get("draft_id"), int(body.get("revision", 0))
        )
    except (
        schema_service.SchemaRevisionConflict,
        schema_service.SchemaOperationError,
    ) as exc:
        return _failure(exc)
    return JsonResponse(result)


@login_required
@require_http_methods(["GET"])
def schema_diff(request, col_id: int):
    collection = _collection(col_id)
    if denied := require_view(collection, request.user):
        return denied
    try:
        return JsonResponse(schema_service.draft_diff(collection))
    except schema_service.SchemaOperationError as exc:
        return error_response(exc)


@login_required
@require_http_methods(["POST"])
def schema_publish(request, col_id: int):
    collection = _collection(col_id)
    if denied := require_manage(collection, request.user):
        return denied
    try:
        schema_service.publish_draft(
            collection,
            request.user,
            load_body(request),
            parse_revision(request),
        )
    except (
        schema_service.SchemaRevisionConflict,
        schema_service.SchemaOperationError,
    ) as exc:
        return _failure(exc)
    return JsonResponse(workspace_envelope(collection, request.user))


@login_required
@require_http_methods(["POST"])
def schema_discard(request, col_id: int):
    collection = _collection(col_id)
    if denied := require_manage(collection, request.user):
        return denied
    try:
        body = load_body(request)
        revision = matching_body_revision(request, body, "revision")
        schema_service.discard_draft(collection, body.get("draft_id"), revision)
    except (
        schema_service.SchemaRevisionConflict,
        schema_service.SchemaOperationError,
    ) as exc:
        return _failure(exc)
    return JsonResponse(workspace_envelope(collection, request.user))


@login_required
@require_http_methods(["GET"])
def schema_versions(request, col_id: int):
    collection = _collection(col_id)
    if denied := require_view(collection, request.user):
        return denied
    try:
        return JsonResponse(
            schema_service.history_page(collection, request.GET.get("cursor"))
        )
    except schema_service.SchemaOperationError as exc:
        return error_response(exc)


@login_required
@require_http_methods(["GET"])
def schema_version_diff(request, col_id: int, version_id: int):
    collection = _collection(col_id)
    if denied := require_view(collection, request.user):
        return denied
    try:
        return JsonResponse(schema_service.version_diff(collection, version_id))
    except schema_service.SchemaOperationError as exc:
        return error_response(exc)


@login_required
@require_http_methods(["POST"])
def schema_restore(request, col_id: int, version_id: int):
    collection = _collection(col_id)
    if denied := require_manage(collection, request.user):
        return denied
    try:
        challenge = schema_service.restore_version(collection, request.user, version_id)
    except schema_service.SchemaOperationError as exc:
        return error_response(exc)
    if challenge is not None:
        return JsonResponse(challenge, status=409)
    return JsonResponse(workspace_envelope(collection, request.user))


@login_required
@require_http_methods(["POST"])
def schema_restore_replace(request, col_id: int):
    collection = _collection(col_id)
    if denied := require_manage(collection, request.user):
        return denied
    try:
        body = load_body(request)
        revision = matching_body_revision(request, body, "existing_draft_revision")
        schema_service.replace_with_version(
            collection,
            request.user,
            int(body.get("version_id", 0)),
            str(body.get("challenge_token", "")),
            revision,
        )
    except (
        schema_service.SchemaRevisionConflict,
        schema_service.SchemaOperationError,
    ) as exc:
        return _failure(exc)
    return JsonResponse(workspace_envelope(collection, request.user))
