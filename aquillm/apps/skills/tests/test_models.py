"""Skill model tests."""
import pytest
from django.contrib.auth import get_user_model
from django.db import IntegrityError

from apps.skills.models import Skill

User = get_user_model()


@pytest.fixture
def user(db):
    return User.objects.create_user(username="alice", password="pw")


@pytest.mark.django_db
def test_create_skill(user):
    s = Skill.objects.create(user=user, name="Pirate", body="Reply like a pirate.")
    assert s.id is not None
    assert s.enabled is True
    assert s.created_at is not None
    assert s.updated_at is not None


@pytest.mark.django_db
def test_unique_per_user(user):
    Skill.objects.create(user=user, name="Pirate", body="x")
    with pytest.raises(IntegrityError):
        Skill.objects.create(user=user, name="Pirate", body="y")


@pytest.mark.django_db
def test_same_name_different_users(db, user):
    other = User.objects.create_user(username="bob", password="pw")
    Skill.objects.create(user=user, name="Pirate", body="x")
    Skill.objects.create(user=other, name="Pirate", body="y")  # no error
