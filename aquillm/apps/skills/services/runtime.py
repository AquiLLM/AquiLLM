"""Per-user DB skill body loader (merged into chat system prompt).

Provides both a sync (`load_user_skill_bodies`) and an async
(`aload_user_skill_bodies`) entry point so callers can pick the right one
for their context — chat consumers run inside an event loop.
"""
from __future__ import annotations

from channels.db import database_sync_to_async
from django.conf import settings

from apps.skills.models import Skill


_SECTION_SEP = "\n\n---\n\n"


def _format_skills(skills) -> str:
    parts: list[str] = []
    for skill in skills:
        body = (skill.body or "").strip()
        if not body:
            continue
        parts.append(f"## {skill.name}\n\n{body}")
    if not parts:
        return ""
    return _SECTION_SEP.join(parts)


def _fetch_user_skills(user_id: int):
    return list(Skill.objects.filter(user_id=user_id, enabled=True).order_by("name"))


_afetch_user_skills = database_sync_to_async(_fetch_user_skills)


def load_user_skill_bodies(user_id: int) -> str:
    """Sync loader. Use from management commands, tests, sync views."""
    if not getattr(settings, "SKILLS_ENABLED", False):
        return ""
    return _format_skills(_fetch_user_skills(user_id))


async def aload_user_skill_bodies(user_id: int) -> str:
    """Async loader. Use from ASGI consumers / async views."""
    if not getattr(settings, "SKILLS_ENABLED", False):
        return ""
    return _format_skills(await _afetch_user_skills(user_id))


__all__ = ["load_user_skill_bodies", "aload_user_skill_bodies"]
