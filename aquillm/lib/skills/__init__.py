"""
Pluggable runtime skills: Python modules that expose `get_tools` and optional
`get_system_prompt_extra` against a `SkillRuntimeContext` (no ORM objects).
"""
from __future__ import annotations

from .base import resolve_modules
from .loader import DEFAULT_BUILTIN_MODULES, iter_all_module_names, load_modules
from .markdown import load_markdown_prompt_bodies
from .registry import collect_system_prompt_extras, collect_tools
from .types import SkillModule, SkillRuntimeContext

__all__ = [
    "DEFAULT_BUILTIN_MODULES",
    "SkillModule",
    "SkillRuntimeContext",
    "collect_system_prompt_extras",
    "collect_tools",
    "iter_all_module_names",
    "load_markdown_prompt_bodies",
    "load_modules",
    "resolve_modules",
]
