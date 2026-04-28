"""Tests for markdown SKILL.md loader."""
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from lib.skills.loader import build_skills_prompt_extra, load_skills


@pytest.fixture
def skills_dir(tmp_path: Path) -> Path:
    """Create a fake skills dir with one valid skill, one bare skill, one bogus."""
    a = tmp_path / "alpha"
    a.mkdir()
    (a / "SKILL.md").write_text(
        "---\nname: alpha\ndescription: Alpha skill\n---\n\nAlpha body.\n"
    )

    b = tmp_path / "bare"
    b.mkdir()
    (b / "SKILL.md").write_text("Body without frontmatter.\n")

    bogus = tmp_path / "broken"
    bogus.mkdir()
    (bogus / "SKILL.md").write_text("---\nname: broken\nno terminator")

    nodir = tmp_path / "nofile"
    nodir.mkdir()  # no SKILL.md inside

    return tmp_path


class TestLoadSkills:
    def test_disabled_returns_empty(self, skills_dir: Path):
        with patch.dict(os.environ, {"SKILLS_ENABLED": "0", "SKILLS_DIR": str(skills_dir)}):
            assert load_skills() == []

    def test_no_dir_returns_empty(self):
        with patch.dict(os.environ, {"SKILLS_ENABLED": "1", "SKILLS_DIR": ""}):
            assert load_skills() == []

    def test_missing_dir_returns_empty(self, tmp_path: Path):
        bad = tmp_path / "does_not_exist"
        with patch.dict(os.environ, {"SKILLS_ENABLED": "1", "SKILLS_DIR": str(bad)}):
            assert load_skills() == []

    def test_loads_valid_and_bare_skills(self, skills_dir: Path):
        with patch.dict(os.environ, {"SKILLS_ENABLED": "1", "SKILLS_DIR": str(skills_dir)}):
            skills = load_skills()
            names = sorted(s.name for s in skills)
            # alpha (frontmatter), bare (no frontmatter, name=dir). broken=skipped.
            assert names == ["alpha", "bare"]

    def test_skill_body_and_meta(self, skills_dir: Path):
        with patch.dict(os.environ, {"SKILLS_ENABLED": "1", "SKILLS_DIR": str(skills_dir)}):
            skills = {s.name: s for s in load_skills()}
            assert skills["alpha"].description == "Alpha skill"
            assert "Alpha body." in skills["alpha"].body
            assert skills["bare"].description == ""
            assert "Body without frontmatter." in skills["bare"].body


class TestBuildSkillsPromptExtra:
    def test_empty(self):
        assert build_skills_prompt_extra([]) == ""

    def test_index_only_no_bodies(self, skills_dir: Path):
        """Progressive disclosure: index lists names + descriptions, never bodies."""
        with patch.dict(os.environ, {"SKILLS_ENABLED": "1", "SKILLS_DIR": str(skills_dir)}):
            extra = build_skills_prompt_extra(load_skills())
            assert "alpha" in extra
            assert "Alpha skill" in extra  # description
            assert "Alpha body." not in extra  # body must NOT leak
            assert "load_skill" in extra  # mentions the loader tool
