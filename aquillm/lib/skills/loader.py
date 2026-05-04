"""Load skill modules from dotted import paths (builtin + settings-driven extras)."""
from __future__ import annotations

import importlib
import structlog
from types import ModuleType
from typing import Iterator, Sequence

_logger = structlog.stdlib.get_logger(__name__)

# `example_runtime_skill` ships in prod when skills are enabled. For a line-by-line
# template, see `lib.skills/builtin/dummy_skill.py` (load via AQUILLM_SKILLS_EXTRA_MODULES).
DEFAULT_BUILTIN_MODULES: tuple[str, ...] = ("lib.skills.builtin.example_runtime_skill",)


def load_modules(paths: Sequence[str]) -> list[ModuleType]:
    """
    Import each path; log and skip modules that cannot be loaded.

    Order is preserved so registry merge ordering stays deterministic.
    """
    out: list[ModuleType] = []
    for dotted in paths:
        try:
            out.append(importlib.import_module(dotted))
        except Exception as exc:  # noqa: BLE001 — show misconfiguration without breaking chat
            _logger.warning("skill_module_import_failed", path=dotted, error=str(exc))
    return out


def iter_all_module_names(
    extra: Sequence[str] | None = None,
) -> Iterator[str]:
    for name in DEFAULT_BUILTIN_MODULES:
        yield name
    if not extra:
        return
    for name in extra:
        if name and name not in DEFAULT_BUILTIN_MODULES:
            yield name


__all__ = [
    "DEFAULT_BUILTIN_MODULES",
    "iter_all_module_names",
    "load_modules",
]
