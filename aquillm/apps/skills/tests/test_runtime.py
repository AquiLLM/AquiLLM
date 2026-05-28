"""Tests for per-user DB skill body loader."""
import pytest
from django.contrib.auth import get_user_model
from django.test import override_settings

from apps.skills.models import Skill
from apps.skills.services.runtime import load_user_skill_bodies

User = get_user_model()


@pytest.fixture
def user(db):
    return User.objects.create_user(username="alice", password="pw")


@pytest.mark.django_db
@override_settings(SKILLS_ENABLED=True)
def test_returns_empty_when_no_skills(user):
    assert load_user_skill_bodies(user.id) == ""


@pytest.mark.django_db
@override_settings(SKILLS_ENABLED=False)
def test_returns_empty_when_feature_disabled(user):
    Skill.objects.create(user=user, name="Pirate", body="Arrr.")
    assert load_user_skill_bodies(user.id) == ""


@pytest.mark.django_db
@override_settings(SKILLS_ENABLED=True)
def test_only_enabled_skills_included(user):
    Skill.objects.create(user=user, name="On", body="visible", enabled=True)
    Skill.objects.create(user=user, name="Off", body="hidden", enabled=False)
    out = load_user_skill_bodies(user.id)
    assert "visible" in out
    assert "hidden" not in out


@pytest.mark.django_db
@override_settings(SKILLS_ENABLED=True)
def test_format_matches_markdown_loader_shape(user):
    Skill.objects.create(user=user, name="A", body="alpha")
    Skill.objects.create(user=user, name="B", body="beta")
    out = load_user_skill_bodies(user.id)
    # Section headings with ## and `---` separator (mirrors lib.skills.markdown).
    assert "## A" in out
    assert "## B" in out
    assert "\n\n---\n\n" in out


@pytest.mark.django_db
@override_settings(SKILLS_ENABLED=True)
def test_isolated_per_user(db, user):
    other = User.objects.create_user(username="bob", password="pw")
    Skill.objects.create(user=user, name="MySkill", body="alice-only")
    assert "alice-only" in load_user_skill_bodies(user.id)
    assert load_user_skill_bodies(other.id) == ""
