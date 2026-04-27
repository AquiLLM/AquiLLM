"""Types for pluggable AquiLLM runtime skills (tools + optional prompt text)."""
from __future__ import annotations

from typing import NotRequired, Protocol, TypedDict, runtime_checkable

from lib.llm.types import LLMTool


class SkillRuntimeContext(TypedDict, total=False):
    """ORM-free context passed to skills. Keep values JSON-friendly."""

    user_id: int
    username: str
    conversation_id: NotRequired[int]


@runtime_checkable
class SkillModule(Protocol):
    """
    A Python module that implements a runtime skill.

    Required: `get_tools(ctx) -> list[LLMTool]`. Optional: `get_system_prompt_extra(ctx) -> str`.
    """

    def get_tools(self, ctx: SkillRuntimeContext) -> list[LLMTool]:
        """Return additional LLM tools for this request context."""
        ...


__all__ = ["SkillModule", "SkillRuntimeContext"]
