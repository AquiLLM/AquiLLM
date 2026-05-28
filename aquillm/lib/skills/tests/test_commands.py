"""Tests for slash command parsing + skill activation."""
from pathlib import Path

from lib.skills.commands import (
    find_skill_for_command,
    format_activated_skill_block,
    parse_slash_command,
)
from lib.skills.loader import Skill


def _skill(name: str, body: str = "BODY") -> Skill:
    return Skill(name=name, description="", body=body, path=Path("/fake"))


class TestParseSlashCommand:
    def test_no_slash(self):
        assert parse_slash_command("hello world") is None

    def test_empty(self):
        assert parse_slash_command("") is None

    def test_only_slash(self):
        assert parse_slash_command("/") is None

    def test_simple_command(self):
        assert parse_slash_command("/caveman") == ("caveman", "")

    def test_command_with_args(self):
        assert parse_slash_command("/caveman ultra") == ("caveman", "ultra")

    def test_command_with_multiword_args(self):
        assert parse_slash_command("/caveman lite mode please") == (
            "caveman",
            "lite mode please",
        )

    def test_strips_outer_whitespace(self):
        assert parse_slash_command("   /caveman ultra  ") == ("caveman", "ultra")

    def test_hyphen_underscore_in_name(self):
        assert parse_slash_command("/old-english") == ("old-english", "")
        assert parse_slash_command("/snake_case") == ("snake_case", "")

    def test_slash_in_middle_no_match(self):
        assert parse_slash_command("hi /caveman ultra") is None


class TestFindSkillForCommand:
    def test_exact_match(self):
        skills = [_skill("caveman"), _skill("other")]
        assert find_skill_for_command(skills, "caveman").name == "caveman"

    def test_case_insensitive(self):
        skills = [_skill("Caveman")]
        assert find_skill_for_command(skills, "CAVEMAN").name == "Caveman"

    def test_no_match(self):
        skills = [_skill("caveman")]
        assert find_skill_for_command(skills, "ghost") is None

    def test_empty_skill_list(self):
        assert find_skill_for_command([], "anything") is None


class TestFormatActivatedSkillBlock:
    def test_no_args(self):
        block = format_activated_skill_block(_skill("caveman", body="cave body"), "")
        assert "# Activated Skill: caveman" in block
        assert "cave body" in block
        assert "Arguments" not in block

    def test_with_args(self):
        block = format_activated_skill_block(_skill("caveman", body="cave body"), "ultra")
        assert "Arguments from /caveman: `ultra`" in block
        assert "cave body" in block
