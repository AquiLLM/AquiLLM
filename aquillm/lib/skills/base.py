"""Skill loading helpers shared by tests and the Django integration layer."""
from __future__ import annotations

from types import ModuleType
from typing import Sequence

from lib.skills.loader import iter_all_module_names, load_modules


def resolve_modules(extra_paths: Sequence[str] | None) -> list[ModuleType]:
    """Return loaded skill modules in deterministic order (builtin first, then extras)."""
    names = list(iter_all_module_names(extra_paths))
    return load_modules(names)


__all__ = ["resolve_modules"]
