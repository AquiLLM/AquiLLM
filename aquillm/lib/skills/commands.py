"""Slash command shortcut for skills.

If the latest user message starts with `/<word>` and `<word>` matches a loaded
skill name, the skill's full body is treated as activated for the rest of the
conversation: it gets appended to the consumer's effective system prompt.
"""
from __future__ import annotations

import re

from lib.skills.loader import Skill

# /word at start, optional args after. word: letters/digits/_/-
_SLASH_RE = re.compile(r"^/([A-Za-z0-9_\-]+)(?:\s+(.*))?\s*$", re.DOTALL)


def parse_slash_command(text: str) -> tuple[str, str] | None:
    """Return (skill_name, args) if text starts with a slash command, else None."""
    if not text:
        return None
    match = _SLASH_RE.match(text.strip())
    if not match:
        return None
    return match.group(1), (match.group(2) or "").strip()


def find_skill_for_command(
    skills: list[Skill], command_name: str
) -> Skill | None:
    """Case-insensitive lookup of a skill by command name."""
    target = command_name.lower()
    for skill in skills:
        if skill.name.lower() == target:
            return skill
    return None


def format_activated_skill_block(skill: Skill, args: str) -> str:
    """Build the system-prompt block injected when a skill is slash-activated."""
    header = f"# Activated Skill: {skill.name}"
    if args:
        header += f"\n\n_Arguments from /{skill.name}: `{args}`_"
    return f"{header}\n\n{skill.body}"
