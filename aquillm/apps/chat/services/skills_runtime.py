"""Django bridge for DB-backed user skills (the Monaco-editor feature).

Joins the per-user `Skill` table bodies onto the conversation's base
system prompt so they reach the LLM at memory-augmentation time. The
file-based skill system (`lib.skills`, progressive-disclosure SKILL.md
packs) is wired separately in `apps.chat.consumers.chat` via the
`load_skill` / `read_skill_file` tools — that path does NOT go through
this module.
"""
from __future__ import annotations

from typing import Any

import structlog
from django.conf import settings

from apps.skills.services.runtime import (
    aload_user_skill_bodies,
    load_user_skill_bodies,
)

logger = structlog.stdlib.get_logger(__name__)


def _compose_base_system(base: str, db_extra: str) -> str:
    if not db_extra:
        return base
    return f"{base.rstrip()}\n\n{db_extra}"


def effective_base_system_for_memory(consumer: Any) -> str:
    """Sync variant. Per-collection notes intentionally excluded — they live
    in tool results (see `documents.py::_inject_collection_notes`); putting
    them in the system prompt makes gpt-4o skip vector_search."""
    if consumer.db_convo is None:
        return ""
    base = consumer.db_convo.system_prompt or ""
    if not getattr(settings, "SKILLS_ENABLED", False):
        return base
    db_extra = load_user_skill_bodies(consumer.user.id).strip()
    return _compose_base_system(base, db_extra)


async def aeffective_base_system_for_memory(consumer: Any) -> str:
    """Async variant — must be awaited from ASGI / consumer code."""
    if consumer.db_convo is None:
        return ""
    base = consumer.db_convo.system_prompt or ""
    if not getattr(settings, "SKILLS_ENABLED", False):
        return base
    db_extra = (await aload_user_skill_bodies(consumer.user.id)).strip()
    return _compose_base_system(base, db_extra)


__all__ = [
    "aeffective_base_system_for_memory",
    "effective_base_system_for_memory",
]
