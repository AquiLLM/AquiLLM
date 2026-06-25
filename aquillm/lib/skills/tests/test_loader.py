"""Loader imports."""
from __future__ import annotations

from lib.skills.loader import iter_all_module_names, load_modules


def test_iter_all_module_names_builtin_then_extra_no_dup():
    names = list(iter_all_module_names(["lib.skills.builtin.example_runtime_skill", "extra.mod"]))
    assert names[0] == "lib.skills.builtin.example_runtime_skill"
    assert "extra.mod" in names
    assert names.count("lib.skills.builtin.example_runtime_skill") == 1


def test_load_modules_skips_bad_path():
    mods = load_modules(["lib.skills.builtin.example_runtime_skill", "not_a_real_module_xyz123"])
    assert len(mods) == 1
    assert mods[0].__name__ == "lib.skills.builtin.example_runtime_skill"


def test_dummy_skill_get_tools_one_tool():
    from lib.skills.builtin import dummy_skill
    from lib.skills.types import SkillRuntimeContext

    ctx: SkillRuntimeContext = {"user_id": 1, "username": "t"}
    tools = dummy_skill.get_tools(ctx)
    assert len(tools) == 1
    assert tools[0].name == "dummy_template_echo"
