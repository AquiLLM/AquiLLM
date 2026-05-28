"""Skill tools: progressive disclosure entry points.

- load_skill: fetches a named skill's SKILL.md body
- read_skill_file: fetches a referenced text file inside the skill dir,
  scoped to that dir (rejects path traversal)
"""
from __future__ import annotations

from pathlib import Path

from lib.llm.decorators import llm_tool
from lib.llm.types import LLMTool, ToolResultDict
from lib.skills.loader import Skill

# Allowlisted text file extensions for read_skill_file. Scripts intentionally excluded.
_ALLOWED_SUFFIXES = frozenset({".md", ".txt", ".json", ".yaml", ".yml", ".csv", ".rst"})

# Cap returned file size (bytes) to keep context usage bounded.
_MAX_FILE_BYTES = 64 * 1024


def build_load_skill_tool(skills: list[Skill]) -> LLMTool:
    """Create a load_skill tool bound to the loaded-skills list."""
    skills_by_name = {s.name: s for s in skills}
    available = ", ".join(sorted(skills_by_name.keys())) or "<none>"

    @llm_tool(
        for_whom="assistant",
        description=(
            "Load a skill's full instructions (SKILL.md body) into context. "
            "Call this when a skill's description matches the user's intent. "
            "The body may reference subfiles in the skill dir — fetch those "
            "with the read_skill_file tool. "
            f"Available skills: {available}."
        ),
        required=["name"],
        param_descs={"name": "Exact skill name to load."},
    )
    def load_skill(name: str) -> ToolResultDict:
        """Return the full body of the named skill, or an error if unknown."""
        skill = skills_by_name.get(name)
        if skill is None:
            return {
                "exception": (
                    f"Unknown skill {name!r}. Available: {available}."
                )
            }
        return {"result": skill.body}

    return load_skill


def build_read_skill_file_tool(skills: list[Skill]) -> LLMTool:
    """Create a read_skill_file tool scoped to each skill's directory."""
    skill_dirs: dict[str, Path] = {s.name: s.path.parent.resolve() for s in skills}
    available = ", ".join(sorted(skill_dirs.keys())) or "<none>"

    @llm_tool(
        for_whom="assistant",
        description=(
            "Read a referenced text file inside a skill's directory (e.g. "
            "references/X.md mentioned in SKILL.md). Path is relative to the "
            "skill dir; absolute paths and parent-dir traversal are rejected. "
            f"Allowed extensions: {', '.join(sorted(_ALLOWED_SUFFIXES))}. "
            f"Available skills: {available}."
        ),
        required=["name", "path"],
        param_descs={
            "name": "Skill name (must match a loaded skill).",
            "path": "Relative path inside the skill dir, e.g. 'references/ultra.md'.",
        },
    )
    def read_skill_file(name: str, path: str) -> ToolResultDict:
        """Read a text file inside a skill dir. Path-scoped, size-capped."""
        skill_dir = skill_dirs.get(name)
        if skill_dir is None:
            return {"exception": f"Unknown skill {name!r}. Available: {available}."}

        rel = Path(path)
        if rel.is_absolute():
            return {"exception": "path must be relative to the skill directory"}

        candidate = (skill_dir / rel).resolve()
        try:
            candidate.relative_to(skill_dir)
        except ValueError:
            return {"exception": "path escapes skill directory"}

        if not candidate.is_file():
            return {"exception": f"file not found: {path}"}

        if candidate.suffix.lower() not in _ALLOWED_SUFFIXES:
            return {
                "exception": (
                    f"file extension not allowed: {candidate.suffix}. "
                    f"Allowed: {', '.join(sorted(_ALLOWED_SUFFIXES))}"
                )
            }

        try:
            data = candidate.read_bytes()
        except OSError as exc:
            return {"exception": f"read failed: {exc}"}

        if len(data) > _MAX_FILE_BYTES:
            return {
                "exception": (
                    f"file too large ({len(data)} bytes); cap is {_MAX_FILE_BYTES}"
                )
            }

        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            return {"exception": "file is not valid UTF-8 text"}

        return {"result": text}

    return read_skill_file
