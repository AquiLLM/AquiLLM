"""Stub API views for collection schema editor development."""
from __future__ import annotations

from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from django.views.decorators.http import require_http_methods

from apps.collections.models import Collection

from .schema_api_helpers import (
    conflict_response,
    load_body,
    parse_revision,
    require_edit,
    require_manage,
    workspace_envelope,
)


@login_required
@require_http_methods(["GET"])
def schema_workspace(request, col_id: int):
    collection = get_object_or_404(Collection, id=col_id)
    return JsonResponse(workspace_envelope(collection, request.user))


@login_required
@require_http_methods(["POST"])
def schema_create_draft(request, col_id: int):
    collection = get_object_or_404(Collection, id=col_id)
    if denied := require_edit(collection, request.user):
        return denied
    return JsonResponse(workspace_envelope(collection, request.user, revision=1))


@login_required
@require_http_methods(["PUT", "DELETE"])
def schema_entity(request, col_id: int, entity_key: str):
    collection = get_object_or_404(Collection, id=col_id)
    if denied := require_edit(collection, request.user):
        return denied
    revision = parse_revision(request)
    if revision is not None and revision < 5:
        return conflict_response(revision)
    return JsonResponse(workspace_envelope(collection, request.user, revision=(revision or 2) + 1))


@login_required
@require_http_methods(["PUT", "DELETE"])
def schema_relation(request, col_id: int, relation_key: str):
    return schema_entity(request, col_id, relation_key)


@login_required
@require_http_methods(["POST"])
def schema_validate(request, col_id: int):
    collection = get_object_or_404(Collection, id=col_id)
    if denied := require_edit(collection, request.user):
        return denied
    body = load_body(request)
    draft_id = body.get("draft_id", "draft-manage-1")
    revision = int(body.get("revision", 5))
    return JsonResponse(
        {
            "identity": {
                "draft_id": draft_id,
                "revision": revision,
                "candidate_checksum": "candidate-checksum-v5",
                "result_id": "validation-result-1",
            },
            "issues": [
                {
                    "code": "alias_duplicate",
                    "location": "entity.person.aliases",
                    "message": "Alias already exists",
                    "severity": "warning",
                }
            ],
            "diff_summary": {
                "base_version": 4,
                "base_checksum": "pub-edit-checksum",
                "candidate_version": revision,
                "candidate_checksum": "candidate-checksum-v5",
                "entities": {"added": 0, "changed": 1, "removed": 0},
                "relations": {"added": 0, "changed": 0, "removed": 0},
            },
        }
    )


@login_required
@require_http_methods(["GET"])
def schema_diff(request, col_id: int):
    get_object_or_404(Collection, id=col_id)
    return JsonResponse(
        {
            "base_version": 4,
            "base_checksum": "pub-edit-checksum",
            "candidate_version": 5,
            "candidate_checksum": "candidate-checksum-v5",
            "entities": {"added": 0, "changed": 1, "removed": 0},
            "relations": {"added": 0, "changed": 0, "removed": 0},
        }
    )


@login_required
@require_http_methods(["POST"])
def schema_publish(request, col_id: int):
    collection = get_object_or_404(Collection, id=col_id)
    if denied := require_manage(collection, request.user):
        return denied
    envelope = workspace_envelope(collection, request.user)
    envelope["draft"] = None
    envelope["published"]["version"] = 5
    envelope["published"]["checksum"] = "candidate-checksum-v5"
    return JsonResponse(envelope)


@login_required
@require_http_methods(["POST"])
def schema_discard(request, col_id: int):
    collection = get_object_or_404(Collection, id=col_id)
    if denied := require_manage(collection, request.user):
        return denied
    revision = parse_revision(request)
    if revision is not None and revision < 5:
        return conflict_response(revision)
    envelope = workspace_envelope(collection, request.user)
    envelope["draft"] = None
    return JsonResponse(envelope)


@login_required
@require_http_methods(["GET"])
def schema_versions(request, col_id: int):
    get_object_or_404(Collection, id=col_id)
    return JsonResponse(
        {
            "versions": [
                {
                    "version": 4,
                    "checksum": "pub-edit-checksum",
                    "published_at": "2026-08-20T12:00:00Z",
                    "summary": "Published person description baseline",
                },
                {
                    "version": 3,
                    "checksum": "pub-view-checksum",
                    "published_at": "2026-08-19T12:00:00Z",
                    "summary": "Initial inherited schema",
                },
            ],
            "next_cursor": "cursor-v2",
            "has_more": True,
        }
    )


@login_required
@require_http_methods(["GET"])
def schema_version_diff(request, col_id: int, version_id: int):
    get_object_or_404(Collection, id=col_id)
    return JsonResponse(
        {
            "base_version": version_id - 1,
            "base_checksum": "pub-view-checksum",
            "candidate_version": version_id,
            "candidate_checksum": "pub-edit-checksum",
            "entities": {"added": 0, "changed": 1, "removed": 0},
            "relations": {"added": 0, "changed": 0, "removed": 0},
        }
    )


@login_required
@require_http_methods(["POST"])
def schema_restore(request, col_id: int, version_id: int):
    collection = get_object_or_404(Collection, id=col_id)
    if denied := require_manage(collection, request.user):
        return denied
    envelope = workspace_envelope(collection, request.user)
    if envelope.get("draft"):
        return JsonResponse(
            {
                "challenge_token": "restore-challenge-token",
                "existing_draft_revision": envelope["draft"]["revision"],
                "existing_draft_id": envelope["draft"]["draft_id"],
                "last_editor": envelope["draft"]["last_editor"],
            },
            status=409,
        )
    return JsonResponse(workspace_envelope(collection, request.user, revision=1))


@login_required
@require_http_methods(["POST"])
def schema_restore_replace(request, col_id: int):
    collection = get_object_or_404(Collection, id=col_id)
    if denied := require_manage(collection, request.user):
        return denied
    body = load_body(request)
    revision = parse_revision(request) or int(body.get("existing_draft_revision", 0))
    if revision < 5:
        return conflict_response(revision)
    if body.get("challenge_token") != "restore-challenge-token":
        return JsonResponse({"error": "invalid_challenge"}, status=400)
    return JsonResponse(workspace_envelope(collection, request.user, revision=1))
