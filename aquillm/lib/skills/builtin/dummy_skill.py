"""
Dummy / template skill — **not** loaded by default.

Copy this file to a new module under `lib/skills/builtin/your_name.py` (or another
importable package), then:
  1. Set a unique `SKILL_ID` string.
  2. Replace `get_tools` / `get_system_prompt_extra` with your logic.
  3. Add your module path to `AQUILLM_SKILLS_EXTRA_MODULES` in settings/env, **or** add
     it to `DEFAULT_BUILTIN_MODULES` in `lib/skills/loader.py` if it should always ship.
  4. Keep this package free of `apps.*` imports; use `SkillRuntimeContext` for user/convo ids.

To try *this* file locally: set `AQUILLM_SKILLS_EXTRA_MODULES=lib.skills.builtin.dummy_skill`
(along with `AQUILLM_SKILLS_ENABLED=1`).

A production-oriented sample that stays enabled by default lives in
`example_runtime_skill.py` next to this file. For **prompt-only** skills in Markdown, use
`docs/skills/runtime/` and `AQUILLM_SKILLS_MARKDOWN_DIR` (no Python code required).
"""
from __future__ import annotations

from lib.llm.decorators import llm_tool
from lib.llm.types import LLMTool, ToolResultDict

from ..types import SkillRuntimeContext

# Pick a stable id for logs, metrics, and ops; does not have to match the file name.
SKILL_ID = "dummy_skill_template"


@llm_tool(
    for_whom="assistant",
    description="Echoes text for smoke-testing the skills pipeline. Safe to remove in real skills.",
    required=["message"],
    param_descs={
        "message": "Short string to echo back. Used to verify tool calls reach your skill code.",
    },
)
def dummy_template_echo(message: str) -> ToolResultDict:
    """Return the same text so you can see end-to-end tool registration without side effects."""
    return {"result": f"[{SKILL_ID}] echo: {message}"}


def get_tools(_ctx: SkillRuntimeContext) -> list[LLMTool]:
    """Entry point: one tool per use case, or more — all must be `LLMTool` instances."""
    return [dummy_template_echo]


def get_system_prompt_extra(_ctx: SkillRuntimeContext) -> str:
    """
    Optional. Merged with the DB system prompt before memory injection.
    Return "" if this skill has no static instructions, or remove this function entirely.
    """
    return (
        f"Template skill ({SKILL_ID}): a `dummy_template_echo` tool is available for pipeline checks."
    )


__all__ = [
    "SKILL_ID",
    "dummy_template_echo",
    "get_system_prompt_extra",
    "get_tools",
]
