"""Readable collection and prompt-skill choices for the chat picker."""

from __future__ import annotations

from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.http import require_GET

from apps.chat.services.collection_prompt_skills import (
    _SKILL_PACK_COLLECTION_NAMES,
    _name_key,
    _setting_enabled,
    accessible_collections,
    discover_collection_skills,
)


@login_required
@require_GET
def chat_context(request):
    collections = accessible_collections(request.user)
    skills_enabled = bool(
        getattr(settings, "SKILLS_ENABLED", False)
    ) and _setting_enabled("AQUILLM_COLLECTION_MARKDOWN_SKILLS_ENABLED")
    return JsonResponse(
        {
            "collections": [
                {
                    "id": collection.pk,
                    "name": collection.name,
                    "parent": collection.parent_id,
                    "path": collection.get_path(),
                    "is_skill_pack": _name_key(collection.name)
                    in _SKILL_PACK_COLLECTION_NAMES,
                }
                for collection in sorted(
                    collections, key=lambda item: (item.get_path().casefold(), item.pk)
                )
            ],
            "skills_enabled": skills_enabled,
            "skills": discover_collection_skills(request.user, collections)
            if skills_enabled
            else [],
        }
    )
