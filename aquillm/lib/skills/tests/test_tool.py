"""Tests for load_skill + read_skill_file tools (progressive disclosure)."""
from pathlib import Path

import pytest

from lib.skills.loader import Skill
from lib.skills.tool import build_load_skill_tool, build_read_skill_file_tool


def _skill(name: str, body: str = "BODY", desc: str = "", path: Path | None = None) -> Skill:
    return Skill(name=name, description=desc, body=body, path=path or Path("/fake"))


@pytest.fixture
def skill_with_refs(tmp_path: Path) -> Skill:
    """Build an on-disk skill dir with a SKILL.md and references/ subdir."""
    skill_dir = tmp_path / "demo"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: demo\n---\nBody.")
    refs = skill_dir / "references"
    refs.mkdir()
    (refs / "ultra.md").write_text("ultra-mode reference content")
    (refs / "secret.sh").write_text("#!/bin/sh\necho nope")
    return Skill(name="demo", description="", body="Body.", path=skill_dir / "SKILL.md")


class TestBuildLoadSkillTool:
    def test_tool_name(self):
        tool = build_load_skill_tool([_skill("alpha")])
        assert tool.name == "load_skill"

    def test_returns_body_for_known_skill(self):
        tool = build_load_skill_tool([_skill("alpha", body="alpha body content")])
        result = tool(name="alpha")
        assert result == {"result": "alpha body content"}

    def test_unknown_skill_returns_exception(self):
        tool = build_load_skill_tool([_skill("alpha")])
        result = tool(name="zeta")
        assert "exception" in result
        assert "zeta" in result["exception"]

    def test_description_lists_available(self):
        tool = build_load_skill_tool([_skill("alpha"), _skill("beta")])
        desc = tool.llm_definition["description"]
        assert "alpha" in desc
        assert "beta" in desc

    def test_empty_skills_list(self):
        tool = build_load_skill_tool([])
        result = tool(name="anything")
        assert "exception" in result


class TestReadSkillFile:
    def test_reads_allowed_text_file(self, skill_with_refs: Skill):
        tool = build_read_skill_file_tool([skill_with_refs])
        result = tool(name="demo", path="references/ultra.md")
        assert result == {"result": "ultra-mode reference content"}

    def test_unknown_skill(self, skill_with_refs: Skill):
        tool = build_read_skill_file_tool([skill_with_refs])
        result = tool(name="ghost", path="references/ultra.md")
        assert "exception" in result
        assert "ghost" in result["exception"]

    def test_rejects_absolute_path(self, skill_with_refs: Skill):
        tool = build_read_skill_file_tool([skill_with_refs])
        result = tool(name="demo", path="/etc/passwd")
        assert "exception" in result
        assert "relative" in result["exception"]

    def test_rejects_parent_traversal(self, skill_with_refs: Skill):
        tool = build_read_skill_file_tool([skill_with_refs])
        result = tool(name="demo", path="../../../etc/passwd")
        assert "exception" in result
        assert "escapes" in result["exception"]

    def test_rejects_disallowed_extension(self, skill_with_refs: Skill):
        tool = build_read_skill_file_tool([skill_with_refs])
        result = tool(name="demo", path="references/secret.sh")
        assert "exception" in result
        assert "extension" in result["exception"]

    def test_missing_file(self, skill_with_refs: Skill):
        tool = build_read_skill_file_tool([skill_with_refs])
        result = tool(name="demo", path="references/missing.md")
        assert "exception" in result
        assert "not found" in result["exception"]

    def test_size_cap(self, tmp_path: Path):
        skill_dir = tmp_path / "big"
        skill_dir.mkdir()
        big = skill_dir / "huge.md"
        big.write_bytes(b"x" * (64 * 1024 + 1))
        skill = Skill(name="big", description="", body="", path=skill_dir / "SKILL.md")
        tool = build_read_skill_file_tool([skill])
        result = tool(name="big", path="huge.md")
        assert "exception" in result
        assert "too large" in result["exception"]
