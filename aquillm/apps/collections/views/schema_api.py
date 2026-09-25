# ruff: noqa: E402, I001 - generation view imports after shared endpoint helpers
from __future__ import annotations

import structlog
from django.contrib.auth.decorators import login_required
from django.db import transaction as transaction
from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from django.views.decorators.http import require_http_methods

from apps.collections.models import (
    Collection,
    CollectionSchemaDraft as CollectionSchemaDraft,
    CollectionSchemaGenerationRun as CollectionSchemaGenerationRun,
)
from apps.collections.services import schema as schema_service
from apps.collections.services.schema_generation import (
    _locked_collection_source_signature as _locked_collection_source_signature,
)
from apps.collections.tasks.schema_generation import (
    enqueue_schema_generation as enqueue_schema_generation,
)

from .schema_api_helpers import (
    body_nonempty_string,
    body_positive_int,
    body_uuid_string,
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

logger = structlog.stdlib.get_logger(__name__)


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
        body = load_body(request)
        draft_id = body_uuid_string(body, "draft_id", "invalid_draft_id")
        values = None
        if request.method == "PUT":
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
            draft_id=draft_id,
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
        draft_id = body_uuid_string(body, "draft_id", "invalid_draft_id")
        revision = body_positive_int(body, "revision", "invalid_revision")
        result = schema_service.validate_draft(
            collection,
            draft_id,
            revision,
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
        body = load_body(request)
        revision = matching_body_revision(request, body, "revision")
        operation = {
            "draft_id": body_uuid_string(body, "draft_id", "invalid_draft_id"),
            "revision": revision,
            "candidate_checksum": body_nonempty_string(
                body,
                "candidate_checksum",
                "invalid_candidate_checksum",
            ),
            "validation_result_id": body_nonempty_string(
                body,
                "validation_result_id",
                "invalid_validation_result_id",
            ),
        }
        schema_service.publish_draft(
            collection,
            request.user,
            operation,
            revision,
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
        draft_id = body_uuid_string(body, "draft_id", "invalid_draft_id")
        schema_service.discard_draft(collection, draft_id, revision)
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
        version_id = body_positive_int(body, "version_id", "invalid_version_id")
        challenge_token = body_nonempty_string(
            body,
            "challenge_token",
            "invalid_challenge_token",
        )
        schema_service.replace_with_version(
            collection,
            request.user,
            version_id,
            challenge_token,
            revision,
        )
    except (
        schema_service.SchemaRevisionConflict,
        schema_service.SchemaOperationError,
    ) as exc:
        return _failure(exc)
    return JsonResponse(workspace_envelope(collection, request.user))


from .schema_generation_api import (
    _enqueue_generation_safely as _enqueue_generation_safely,
    schema_generate as schema_generate,
    schema_generation_status as schema_generation_status,
)
