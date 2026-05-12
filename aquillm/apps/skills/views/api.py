"""JSON CRUD endpoints for per-user markdown skills."""
from __future__ import annotations

import json

from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.db import IntegrityError
from django.http import HttpRequest, JsonResponse
from django.views.decorators.http import require_http_methods

from apps.skills.models import Skill


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


__all__ = ["skills_list_create", "skill_detail"]
