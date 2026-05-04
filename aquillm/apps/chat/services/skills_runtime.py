"""Django-facing bridge for `lib.skills` (settings + `ChatConsumer` context)."""
from __future__ import annotations

import structlog
from pathlib import Path
from typing import Any

from django.conf import settings

from aquillm.llm import LLMTool
from lib.skills import (
    collect_system_prompt_extras,
    collect_tools,
    resolve_modules,
)
from lib.skills.markdown import load_markdown_prompt_bodies
from lib.skills.types import SkillRuntimeContext

logger = structlog.stdlib.get_logger(__name__)


def _repo_root() -> Path:
    return Path(settings.BASE_DIR).parent


def _markdown_skills_root() -> Path | None:
    if not getattr(settings, "SKILLS_ENABLED", False):
        return None
    raw = (getattr(settings, "AQUILLM_SKILLS_MARKDOWN_DIR", None) or "").strip()
    if not raw:
        return None
    p = Path(raw)
    if p.is_absolute():
        if p.is_dir():
            return p
        logger.warning("markdown_skills_dir_not_a_directory", path=str(p))
        return None
    candidate = (_repo_root() / raw).resolve()
    if candidate.is_dir():
        return candidate
    logger.warning("markdown_skills_dir_not_found", path=str(candidate))
    return None


def _resolved_modules():
    return resolve_modules(settings.AQUILLM_SKILLS_EXTRA_MODULES)


def build_skill_runtime_context(consumer: Any) -> SkillRuntimeContext:
    assert consumer.user is not None
    ctx: SkillRuntimeContext = {
        "user_id": consumer.user.id,
        "username": consumer.user.get_username(),
    }
    if consumer.db_convo is not None:
        ctx["conversation_id"] = consumer.db_convo.id
    return ctx


def build_skill_tools(consumer: Any) -> list[LLMTool]:
    if not getattr(settings, "SKILLS_ENABLED", False):
        return []
    ctx = build_skill_runtime_context(consumer)
    return collect_tools(_resolved_modules(), ctx)


def effective_base_system_for_memory(consumer: Any) -> str:
    """
    Base system string (per-DB) plus optional skill text, before memory/episodic injection.
    """
    if consumer.db_convo is None:
        return ""
    base = consumer.db_convo.system_prompt or ""
    if not getattr(settings, "SKILLS_ENABLED", False):
        return base
    ctx = build_skill_runtime_context(consumer)
    py_extra = collect_system_prompt_extras(_resolved_modules(), ctx).strip()
    md_extra = load_markdown_prompt_bodies(_markdown_skills_root()).strip()
    extra_parts = [s for s in (py_extra, md_extra) if s]
    if not extra_parts:
        return base
    extra = "\n\n---\n\n".join(extra_parts)
    return f"{base.rstrip()}\n\n{extra}"


__all__ = [
    "build_skill_runtime_context",
    "build_skill_tools",
    "effective_base_system_for_memory",
]
