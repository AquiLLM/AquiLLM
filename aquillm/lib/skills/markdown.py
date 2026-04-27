"""
Load prompt-only AquiLLM skills from Markdown files (Cursor/Claude-style .md, server-side).

Files are **instructions merged into the base system prompt**; they do not register LLM tools.
For tools, use Python skills (`lib.skills` modules with `get_tools`).

Conventions (see `docs/skills/runtime/` in the repo):
  - Set `AQUILLM_SKILLS_MARKDOWN_DIR` to a path (repo-relative or absolute) when skills are enabled.
  - Every `**/*.md` is loaded except `README.md` and files starting with `_` (e.g. `_template.md`).

Optional front matter (like Cursor `SKILL.md`):

  ---
  name: Short title for this block
  ---

  Free-form markdown body. If front matter is omitted, the whole file is one block titled from
  the filename stem.
"""
from __future__ import annotations

import structlog
from pathlib import Path

_logger = structlog.stdlib.get_logger(__name__)


def _parse_simple_front_matter_block(raw: str) -> tuple[dict[str, str], str]:
    """
    Parse a leading `---` / `---` block. Between delimiters: `key: value` lines.
    If the file does not start with `---`, returns ({}, full_text).
    """
    text = raw.lstrip("\ufeff")
    if not text.startswith("---"):
        return {}, raw
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, raw
    meta_lines: list[str] = []
    i = 1
    while i < len(lines):
        if lines[i].strip() == "---":
            break
        meta_lines.append(lines[i])
        i += 1
    else:
        return {}, raw
    body = "\n".join(lines[i + 1 :]).lstrip("\n")
    meta: dict[str, str] = {}
    for line in meta_lines:
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        k = key.strip().lower().replace(" ", "_")
        v = val.strip()
        if k and v:
            meta[k] = v
    return meta, body


def _section_title(meta: dict[str, str], path: Path) -> str:
    for key in ("name", "title", "id"):
        if key in meta:
            return meta[key]
    return path.stem.replace("_", " ").replace("-", " ").title()


def load_markdown_prompt_bodies(root: Path | None) -> str:
    """
    Return markdown text to append to the system prompt, or "" if the directory is missing/empty.
    """
    if root is None or not root.is_dir():
        return ""
    parts: list[str] = []
    for md_path in sorted(root.rglob("*.md")):
        if md_path.name == "README.md" or md_path.name.startswith("_"):
            continue
        if not md_path.is_file():
            continue
        try:
            raw = md_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            _logger.warning("markdown_skill_read_failed", path=str(md_path), error=str(exc))
            continue
        meta, body = _parse_simple_front_matter_block(raw)
        if meta or (body.strip() and raw.lstrip("\ufeff").startswith("---")):
            block = body.strip()
            if not block:
                continue
            title = _section_title(meta, md_path)
            parts.append(f"## {title}\n\n{block}")
        else:
            full = raw.strip()
            if not full:
                continue
            title = md_path.stem.replace("_", " ").replace("-", " ").title()
            parts.append(f"## {title}\n\n{full}")
    if not parts:
        return ""
    return "\n\n---\n\n".join(parts)


__all__ = [
    "load_markdown_prompt_bodies",
    "_parse_simple_front_matter_block",
]
