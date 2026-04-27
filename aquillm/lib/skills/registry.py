"""Merge tools and system-prompt extras from one or more skill modules."""
from __future__ import annotations

import structlog
from types import ModuleType
from typing import Sequence

from lib.llm.types import LLMTool

from .types import SkillRuntimeContext

_logger = structlog.stdlib.get_logger(__name__)


def _call_optional(mod: ModuleType, name: str, ctx: SkillRuntimeContext):
    fn = getattr(mod, name, None)
    if fn is None or not callable(fn):
        return None
    return fn(ctx)


def collect_tools(modules: Sequence[ModuleType], ctx: SkillRuntimeContext) -> list[LLMTool]:
    """Collect tools; first module wins for duplicate `LLMTool.name` values."""
    out: list[LLMTool] = []
    seen: set[str] = set()

    for mod in modules:
        raw = _call_optional(mod, "get_tools", ctx)
        if raw is None:
            _logger.warning("skill_module_missing_get_tools", module=getattr(mod, "__name__", str(mod)))
            continue
        if not isinstance(raw, list):
            _logger.warning("skill_get_tools_not_list", module=getattr(mod, "__name__", str(mod)))
            continue
        for item in raw:
            if not isinstance(item, LLMTool):
                _logger.warning("skill_tool_not_llmtool", module=getattr(mod, "__name__", str(mod)))
                continue
            n = item.name
            if n in seen:
                _logger.info(
                    "skill_tool_duplicate_ignored",
                    name=n,
                    module=getattr(mod, "__name__", str(mod)),
                )
                continue
            seen.add(n)
            out.append(item)
    return out


def collect_system_prompt_extras(modules: Sequence[ModuleType], ctx: SkillRuntimeContext) -> str:
    """Concatenate non-empty extras in module order with blank lines between."""
    parts: list[str] = []
    for mod in modules:
        raw = _call_optional(mod, "get_system_prompt_extra", ctx)
        if raw is None:
            continue
        if not isinstance(raw, str):
            _logger.warning(
                "skill_system_prompt_extra_not_str",
                module=getattr(mod, "__name__", str(mod)),
            )
            continue
        chunk = raw.strip()
        if chunk:
            parts.append(chunk)
    return "\n\n".join(parts)


__all__ = ["collect_system_prompt_extras", "collect_tools"]
