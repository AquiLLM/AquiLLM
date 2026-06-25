"""
Example runtime skill: one read-only tool that lists repository guardrail commands.

For a **minimal copy-paste skeleton** (echo tool + commented steps), use
`lib.skills/builtin/dummy_skill.py` instead. This file is the default built-in
when `AQUILLM_SKILLS_ENABLED` is on. Keep `lib` free of `apps.*` imports.
"""
from __future__ import annotations

from lib.llm.decorators import llm_tool
from lib.llm.types import LLMTool, ToolResultDict

from ..types import SkillRuntimeContext

SKILL_ID = "example_runtime_skill"


@llm_tool(
    for_whom="assistant",
    description="Returns the commands used in CI to check file length, import boundaries, and repo hygiene.",
    required=[],
)
def aquillm_local_quality_gates() -> ToolResultDict:
    """List local structural checks; does not run them."""
    text = (
        "Backend guardrails: python scripts/check_file_lengths.py; "
        "python scripts/check_import_boundaries.py; "
        "pwsh -ExecutionPolicy Bypass -File scripts/check_hygiene.ps1. "
        "Run backend tests: cd aquillm && python -m pytest -q --tb=short. "
        "If you touched React: cd ../react && npm run typecheck && npm run build."
    )
    return {"result": text}


def get_tools(_ctx: SkillRuntimeContext) -> list[LLMTool]:
    # Decorated functions are already `LLMTool` instances; do not call them here.
    return [aquillm_local_quality_gates]


def get_system_prompt_extra(_ctx: SkillRuntimeContext) -> str:
    return (
        "A built-in 'aquillm_local_quality_gates' tool can restate the repo's structural "
        "verification commands when discussing code health or pre-PR checks."
    )


__all__ = [
    "SKILL_ID",
    "aquillm_local_quality_gates",
    "get_system_prompt_extra",
    "get_tools",
]
