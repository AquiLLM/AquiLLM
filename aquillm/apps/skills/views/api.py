"""JSON CRUD endpoints for per-user markdown skills."""
from __future__ import annotations

import json

import structlog
from asgiref.sync import sync_to_async
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.db import IntegrityError
from django.http import HttpRequest, JsonResponse
from django.views.decorators.http import require_http_methods

from apps.collections.models import Collection
from apps.skills.models import CollectionSkill, Skill, SkillEditSuggestion
from apps.skills.models.collection_skill import MAX_BODY_LENGTH as COLLECTION_SKILL_MAX_BODY
from apps.skills.services.suggestions import (
    _list_pending_feedback_sync,
    accept_suggestion_sync,
    dismiss_suggestion_sync,
    generate_suggestion,
)

logger = structlog.stdlib.get_logger(__name__)


MAX_NAME_LENGTH = 120
MAX_BODY_LENGTH = 50_000


def _serialize(skill: Skill) -> dict:
    return {
        "id": skill.id,
        "name": skill.name,
        "body": skill.body,
        "enabled": skill.enabled,
        "created_at": skill.created_at.isoformat() if skill.created_at else None,
        "updated_at": skill.updated_at.isoformat() if skill.updated_at else None,
    }


def _parse_payload(request: HttpRequest) -> dict | JsonResponse:
    try:
        data = json.loads(request.body or b"{}")
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    if not isinstance(data, dict):
        return JsonResponse({"error": "Payload must be a JSON object"}, status=400)
    return data


def _validate_skill_fields(data: dict, *, partial: bool = False) -> dict | JsonResponse:
    """Returns cleaned subset of {name, body, enabled} or a JsonResponse error."""
    cleaned: dict = {}

    if "name" in data:
        name = data.get("name")
        if not isinstance(name, str):
            return JsonResponse({"error": "name must be a string"}, status=400)
        name = name.strip()
        if not name:
            return JsonResponse({"error": "name is required"}, status=400)
        if len(name) > MAX_NAME_LENGTH:
            return JsonResponse({"error": f"name too long (max {MAX_NAME_LENGTH})"}, status=400)
        cleaned["name"] = name
    elif not partial:
        return JsonResponse({"error": "name is required"}, status=400)

    if "body" in data:
        body = data.get("body")
        if not isinstance(body, str):
            return JsonResponse({"error": "body must be a string"}, status=400)
        if len(body) > MAX_BODY_LENGTH:
            return JsonResponse({"error": f"body too long (max {MAX_BODY_LENGTH} chars)"}, status=400)
        cleaned["body"] = body
    elif not partial:
        cleaned["body"] = ""

    if "enabled" in data:
        enabled = data.get("enabled")
        if not isinstance(enabled, bool):
            return JsonResponse({"error": "enabled must be a boolean"}, status=400)
        cleaned["enabled"] = enabled

    return cleaned


@login_required
@require_http_methods(["GET", "POST"])
def skills_list_create(request: HttpRequest) -> JsonResponse:
    """GET: list current user's skills. POST: create one."""
    if request.method == "GET":
        skills = Skill.objects.filter(user=request.user).order_by("name")
        return JsonResponse(
            {
                "skills_enabled": bool(getattr(settings, "SKILLS_ENABLED", False)),
                "skills": [_serialize(s) for s in skills],
            }
        )

    # POST
    payload = _parse_payload(request)
    if isinstance(payload, JsonResponse):
        return payload
    cleaned = _validate_skill_fields(payload, partial=False)
    if isinstance(cleaned, JsonResponse):
        return cleaned
    try:
        skill = Skill.objects.create(
            user=request.user,
            name=cleaned["name"],
            body=cleaned.get("body", ""),
            enabled=cleaned.get("enabled", True),
        )
    except IntegrityError:
        return JsonResponse({"error": "A skill with that name already exists"}, status=409)
    return JsonResponse(_serialize(skill), status=201)


@login_required
@require_http_methods(["GET", "PUT", "DELETE"])
def skill_detail(request: HttpRequest, skill_id: int) -> JsonResponse:
    """Read / update / delete one of the current user's skills."""
    try:
        skill = Skill.objects.get(pk=skill_id, user=request.user)
    except Skill.DoesNotExist:
        return JsonResponse({"error": "Not found"}, status=404)

    if request.method == "GET":
        return JsonResponse(_serialize(skill))

    if request.method == "DELETE":
        skill.delete()
        return JsonResponse({"status": "deleted"})

    # PUT
    payload = _parse_payload(request)
    if isinstance(payload, JsonResponse):
        return payload
    cleaned = _validate_skill_fields(payload, partial=True)
    if isinstance(cleaned, JsonResponse):
        return cleaned
    for field, value in cleaned.items():
        setattr(skill, field, value)
    try:
        skill.save()
    except IntegrityError:
        return JsonResponse({"error": "A skill with that name already exists"}, status=409)
    return JsonResponse(_serialize(skill))


def _serialize_collection_skill(cs: CollectionSkill | None, collection: Collection) -> dict:
    if cs is None:
        return {
            "collection_id": collection.id,
            "collection_name": collection.name,
            "body": "",
            "updated_at": None,
            "updated_by": None,
            "exists": False,
            "max_body_length": COLLECTION_SKILL_MAX_BODY,
        }
    return {
        "collection_id": collection.id,
        "collection_name": collection.name,
        "body": cs.body,
        "updated_at": cs.updated_at.isoformat() if cs.updated_at else None,
        "updated_by": cs.updated_by.username if cs.updated_by else None,
        "exists": True,
        "max_body_length": COLLECTION_SKILL_MAX_BODY,
    }


@login_required
@require_http_methods(["GET", "PUT"])
def collection_skill_detail(request: HttpRequest, collection_id: int) -> JsonResponse:
    """Read or upsert the Collection Notes for a single collection.

    Permission rules:
      - GET requires EDIT or higher on the collection.
      - PUT requires MANAGE on the collection.
    PUT is upsert: creates the row if it does not exist, updates otherwise.
    """
    if not getattr(settings, "SKILLS_ENABLED", False):
        return JsonResponse({"error": "Skills feature is disabled"}, status=404)

    try:
        collection = Collection.objects.get(pk=collection_id)
    except Collection.DoesNotExist:
        return JsonResponse({"error": "Collection not found"}, status=404)

    if request.method == "GET":
        if not collection.user_can_edit(request.user):
            return JsonResponse({"error": "Forbidden"}, status=403)
        cs = CollectionSkill.objects.filter(collection=collection).first()
        return JsonResponse(_serialize_collection_skill(cs, collection))

    # PUT
    if not collection.user_can_manage(request.user):
        return JsonResponse({"error": "Forbidden"}, status=403)
    payload = _parse_payload(request)
    if isinstance(payload, JsonResponse):
        return payload
    body = payload.get("body", "")
    if not isinstance(body, str):
        return JsonResponse({"error": "body must be a string"}, status=400)
    if len(body) > COLLECTION_SKILL_MAX_BODY:
        return JsonResponse(
            {"error": f"body too long (max {COLLECTION_SKILL_MAX_BODY} chars)"}, status=400
        )
    cs, _ = CollectionSkill.objects.update_or_create(
        collection=collection,
        defaults={"body": body, "updated_by": request.user},
    )
    return JsonResponse(_serialize_collection_skill(cs, collection))


def _require_manager(request: HttpRequest, collection_id: int):
    """Return (Collection, None) if OK; else (None, JsonResponse error)."""
    if not getattr(settings, "SKILLS_ENABLED", False):
        return None, JsonResponse({"error": "Skills feature is disabled"}, status=404)
    try:
        collection = Collection.objects.get(pk=collection_id)
    except Collection.DoesNotExist:
        return None, JsonResponse({"error": "Collection not found"}, status=404)
    if not collection.user_can_manage(request.user):
        return None, JsonResponse({"error": "Forbidden"}, status=403)
    return collection, None


def _serialize_feedback(msg) -> dict:
    return {
        "message_id": msg.id,
        "message_uuid": str(msg.message_uuid),
        "conversation_id": msg.conversation_id,
        "conversation_name": msg.conversation.name,
        "rating": msg.rating,
        "feedback_text": msg.feedback_text,
        "feedback_submitted_at": (
            msg.feedback_submitted_at.isoformat() if msg.feedback_submitted_at else None
        ),
        "model": msg.model,
        "content_preview": (msg.content or "")[:400],
    }


def _serialize_suggestion(s: SkillEditSuggestion) -> dict:
    return {
        "id": s.id,
        "collection_id": s.collection_id,
        "source_message_id": s.source_message_id,
        "notes_body_at_generation": s.notes_body_at_generation,
        "proposed_body": s.proposed_body,
        "status": s.status,
        "generated_by": s.generated_by.username if s.generated_by else None,
        "resolved_by": s.resolved_by.username if s.resolved_by else None,
        "resolved_at": s.resolved_at.isoformat() if s.resolved_at else None,
        "created_at": s.created_at.isoformat() if s.created_at else None,
    }


@login_required
@require_http_methods(["GET"])
def collection_pending_feedback(request: HttpRequest, collection_id: int) -> JsonResponse:
    """List corrective-feedback messages awaiting a manager decision."""
    collection, err = _require_manager(request, collection_id)
    if err:
        return err
    msgs = _list_pending_feedback_sync(collection.id)
    return JsonResponse({"items": [_serialize_feedback(m) for m in msgs]})


@login_required
@require_http_methods(["GET"])
def collection_suggestions_list(request: HttpRequest, collection_id: int) -> JsonResponse:
    """List pending suggestions for this collection."""
    collection, err = _require_manager(request, collection_id)
    if err:
        return err
    qs = (
        SkillEditSuggestion.objects
        .filter(collection=collection, status=SkillEditSuggestion.STATUS_PENDING)
        .select_related("generated_by")
        .order_by("-created_at")
    )
    return JsonResponse({"items": [_serialize_suggestion(s) for s in qs]})


async def collection_suggestions_generate(request: HttpRequest, collection_id: int) -> JsonResponse:
    """Generate a SkillEditSuggestion from a specific feedback message via LLM."""
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)
    user = await sync_to_async(lambda: request.user)()
    if not user.is_authenticated:
        return JsonResponse({"error": "Unauthorized"}, status=401)
    collection, err = await sync_to_async(_require_manager)(request, collection_id)
    if err:
        return err
    try:
        payload = json.loads(request.body or b"{}")
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    message_id = payload.get("message_id")
    if not isinstance(message_id, int):
        return JsonResponse({"error": "message_id (int) is required"}, status=400)
    try:
        suggestion = await generate_suggestion(
            collection_id=collection.id, message_id=message_id, user=user,
        )
    except ValueError as e:
        return JsonResponse({"error": str(e)}, status=400)
    except Exception as e:
        logger.exception("suggestion_generation_failed", error=str(e))
        return JsonResponse({"error": "Generation failed"}, status=500)
    # Serialize on a sync thread so FK lookups (e.g. generated_by.username)
    # don't trip Django's SynchronousOnlyOperation guard.
    payload = await sync_to_async(_serialize_suggestion)(suggestion)
    return JsonResponse(payload, status=201)


@login_required
@require_http_methods(["POST"])
def suggestion_accept(request: HttpRequest, suggestion_id: int) -> JsonResponse:
    """Accept a suggestion, applying its body (or owner-edited override) to the notes."""
    if not getattr(settings, "SKILLS_ENABLED", False):
        return JsonResponse({"error": "Skills feature is disabled"}, status=404)
    try:
        suggestion = (
            SkillEditSuggestion.objects.select_related("collection").get(pk=suggestion_id)
        )
    except SkillEditSuggestion.DoesNotExist:
        return JsonResponse({"error": "Suggestion not found"}, status=404)
    if not suggestion.collection.user_can_manage(request.user):
        return JsonResponse({"error": "Forbidden"}, status=403)
    try:
        payload = json.loads(request.body or b"{}")
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    override_body = payload.get("body")
    if override_body is not None and not isinstance(override_body, str):
        return JsonResponse({"error": "body must be a string"}, status=400)
    if isinstance(override_body, str) and len(override_body) > COLLECTION_SKILL_MAX_BODY:
        return JsonResponse(
            {"error": f"body too long (max {COLLECTION_SKILL_MAX_BODY} chars)"}, status=400
        )
    try:
        accept_suggestion_sync(
            suggestion=suggestion, override_body=override_body, user=request.user,
        )
    except ValueError as e:
        return JsonResponse({"error": str(e)}, status=409)
    suggestion.refresh_from_db()
    return JsonResponse(_serialize_suggestion(suggestion))


@login_required
@require_http_methods(["POST"])
def suggestion_dismiss(request: HttpRequest, suggestion_id: int) -> JsonResponse:
    """Mark a suggestion dismissed without applying it."""
    if not getattr(settings, "SKILLS_ENABLED", False):
        return JsonResponse({"error": "Skills feature is disabled"}, status=404)
    try:
        suggestion = (
            SkillEditSuggestion.objects.select_related("collection").get(pk=suggestion_id)
        )
    except SkillEditSuggestion.DoesNotExist:
        return JsonResponse({"error": "Suggestion not found"}, status=404)
    if not suggestion.collection.user_can_manage(request.user):
        return JsonResponse({"error": "Forbidden"}, status=403)
    try:
        dismiss_suggestion_sync(suggestion=suggestion, user=request.user)
    except ValueError as e:
        return JsonResponse({"error": str(e)}, status=409)
    suggestion.refresh_from_db()
    return JsonResponse(_serialize_suggestion(suggestion))


__all__ = [
    "skills_list_create",
    "skill_detail",
    "collection_skill_detail",
    "collection_pending_feedback",
    "collection_suggestions_list",
    "collection_suggestions_generate",
    "suggestion_accept",
    "suggestion_dismiss",
]
