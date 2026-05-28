"""Load Claude Code-style skills from a directory.

A skill is a subdirectory containing a SKILL.md with YAML frontmatter:

    ---
    name: my-skill
    description: Short blurb the LLM uses to decide when to apply.
    ---

    Markdown body. Becomes part of system prompt.

Skill discovery is recursive one level: SKILLS_DIR contains skill dirs.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import structlog
import yaml

logger = structlog.stdlib.get_logger(__name__)


@dataclass
class Skill:
    """Parsed SKILL.md content."""

    name: str
    description: str
    body: str
    path: Path


def _parse_skill_md(path: Path) -> Skill | None:
    """Read SKILL.md, split frontmatter from body. Returns None on parse failure."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("skill_read_failed", path=str(path), error=str(exc))
        return None

    if not text.startswith("---"):
        # No frontmatter: treat whole file as body, derive name from dir
        return Skill(
            name=path.parent.name,
            description="",
            body=text.strip(),
            path=path,
        )

    parts = text.split("---", 2)
    if len(parts) < 3:
        logger.warning("skill_frontmatter_unterminated", path=str(path))
        return None

    _, frontmatter_text, body = parts
    try:
        meta = yaml.safe_load(frontmatter_text) or {}
    except yaml.YAMLError as exc:
        logger.warning("skill_frontmatter_invalid", path=str(path), error=str(exc))
        return None

    name = str(meta.get("name") or path.parent.name)
    description = str(meta.get("description") or "")
    return Skill(
        name=name,
        description=description,
        body=body.strip(),
        path=path,
    )


def load_skills() -> list[Skill]:
    """Discover skills in SKILLS_DIR. Each subdir with a SKILL.md is one skill."""
    if not os.environ.get("SKILLS_ENABLED", "").strip().lower() in ("1", "true", "yes"):
        return []

    base_dir = os.environ.get("SKILLS_DIR", "").strip()
    if not base_dir:
        return []

    base = Path(base_dir)
    if not base.is_dir():
        logger.warning("skills_dir_missing", path=str(base))
        return []

    skills: list[Skill] = []
    for skill_dir in sorted(base.iterdir()):
        if not skill_dir.is_dir():
            continue
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.is_file():
            continue
        skill = _parse_skill_md(skill_md)
        if skill is None:
            continue
        skills.append(skill)
        logger.info("skill_loaded", name=skill.name, path=str(skill_md))
    return skills


def build_skills_prompt_extra(skills: list[Skill]) -> str:
    """Compose a skill INDEX (name + description) for the system prompt.

    Bodies are NOT included — they're loaded on demand via the load_skill tool
    (progressive disclosure). The LLM uses descriptions to decide when to load
    a skill's full body.
    """
    if not skills:
        return ""
    parts: list[str] = [
        "# Available Skills",
        "",
        "You have access to the following skills. Each is a packaged set of "
        "instructions you can load into your context using the `load_skill` "
        "tool when its description matches the user's intent.",
        "",
    ]
    for skill in skills:
        desc = skill.description or "(no description)"
        parts.append(f"- **{skill.name}** — {desc}")
    return "\n".join(parts).rstrip()
