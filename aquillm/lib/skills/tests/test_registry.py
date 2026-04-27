"""Registry merge behavior."""
from __future__ import annotations

from types import ModuleType

from lib.llm.types import LLMTool
from lib.skills.registry import collect_system_prompt_extras, collect_tools
from lib.skills.types import SkillRuntimeContext


def _dummy_tool(name: str) -> LLMTool:
    def _fn() -> dict:
        return {"result": "ok"}

    return LLMTool(
        llm_definition={"name": name, "description": "d", "parameters": {"type": "object"}},
        for_whom="assistant",
        _function=_fn,
    )


def test_collect_tools_first_wins_on_name_collision():
    a = ModuleType("skill_a")

    def get_tools_a(_ctx: SkillRuntimeContext):
        return [_dummy_tool("same"), _dummy_tool("only_a")]

    a.get_tools = get_tools_a

    b = ModuleType("skill_b")

    def get_tools_b(_ctx: SkillRuntimeContext):
        return [_dummy_tool("same"), _dummy_tool("only_b")]

    b.get_tools = get_tools_b

    ctx: SkillRuntimeContext = {"user_id": 1, "username": "u"}
    tools = collect_tools([a, b], ctx)
    names = [t.name for t in tools]
    assert names == ["same", "only_a", "only_b"]


def test_collect_prompt_extras_joins_non_empty():
    m = ModuleType("skill_c")
    m.get_system_prompt_extra = lambda _ctx: "first"
    n = ModuleType("skill_d")
    n.get_system_prompt_extra = lambda _ctx: "second"
    ctx: SkillRuntimeContext = {"user_id": 1, "username": "u"}
    text = collect_system_prompt_extras([m, n], ctx)
    assert "first" in text and "second" in text
