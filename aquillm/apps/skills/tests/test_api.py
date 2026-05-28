"""HTTP tests for skill CRUD endpoints (auth + ownership scoping)."""
import json

import pytest
from django.contrib.auth import get_user_model
from django.test import Client

from apps.skills.models import Skill

User = get_user_model()


@pytest.fixture
def alice(db):
    return User.objects.create_user(username="alice", password="pw")


@pytest.fixture
def bob(db):
    return User.objects.create_user(username="bob", password="pw")


@pytest.fixture
def alice_client(alice):
    c = Client()
    c.force_login(alice)
    return c


@pytest.fixture
def bob_client(bob):
    c = Client()
    c.force_login(bob)
    return c


# ---- auth -----------------------------------------------------------------


@pytest.mark.django_db
def test_unauthenticated_list_redirects_or_forbids():
    r = Client().get("/api/skills/")
    # login_required redirects (302) or returns 403 depending on settings;
    # either way it must not return 200.
    assert r.status_code in (302, 401, 403)


# ---- list / create --------------------------------------------------------


@pytest.mark.django_db
def test_list_empty(alice_client):
    r = alice_client.get("/api/skills/")
    assert r.status_code == 200
    data = r.json()
    assert data["skills"] == []
    assert "skills_enabled" in data


@pytest.mark.django_db
def test_create_skill(alice_client, alice):
    r = alice_client.post(
        "/api/skills/",
        data=json.dumps({"name": "Pirate", "body": "Reply like a pirate."}),
        content_type="application/json",
    )
    assert r.status_code == 201, r.content
    data = r.json()
    assert data["name"] == "Pirate"
    assert data["enabled"] is True
    assert Skill.objects.filter(user=alice, name="Pirate").exists()


@pytest.mark.django_db
def test_create_rejects_missing_name(alice_client):
    r = alice_client.post(
        "/api/skills/", data=json.dumps({"body": "x"}), content_type="application/json"
    )
    assert r.status_code == 400


@pytest.mark.django_db
def test_create_rejects_duplicate_name(alice_client):
    payload = json.dumps({"name": "Pirate", "body": "x"})
    r1 = alice_client.post("/api/skills/", data=payload, content_type="application/json")
    assert r1.status_code == 201
    r2 = alice_client.post("/api/skills/", data=payload, content_type="application/json")
    assert r2.status_code == 409


@pytest.mark.django_db
def test_create_rejects_oversized_body(alice_client):
    body = "x" * 50_001
    r = alice_client.post(
        "/api/skills/",
        data=json.dumps({"name": "Big", "body": body}),
        content_type="application/json",
    )
    assert r.status_code == 400


# ---- ownership scoping ----------------------------------------------------


@pytest.mark.django_db
def test_list_scoped_to_user(alice_client, bob_client, alice, bob):
    Skill.objects.create(user=alice, name="A", body="alpha")
    Skill.objects.create(user=bob, name="B", body="beta")

    r = alice_client.get("/api/skills/")
    names = [s["name"] for s in r.json()["skills"]]
    assert names == ["A"]

    r2 = bob_client.get("/api/skills/")
    names2 = [s["name"] for s in r2.json()["skills"]]
    assert names2 == ["B"]


@pytest.mark.django_db
def test_cannot_read_other_users_skill(alice_client, bob):
    other = Skill.objects.create(user=bob, name="Secret", body="s")
    r = alice_client.get(f"/api/skills/{other.id}/")
    assert r.status_code == 404


@pytest.mark.django_db
def test_cannot_update_other_users_skill(alice_client, bob):
    other = Skill.objects.create(user=bob, name="Secret", body="s")
    r = alice_client.put(
        f"/api/skills/{other.id}/",
        data=json.dumps({"body": "hacked"}),
        content_type="application/json",
    )
    assert r.status_code == 404
    other.refresh_from_db()
    assert other.body == "s"


@pytest.mark.django_db
def test_cannot_delete_other_users_skill(alice_client, bob):
    other = Skill.objects.create(user=bob, name="Secret", body="s")
    r = alice_client.delete(f"/api/skills/{other.id}/")
    assert r.status_code == 404
    assert Skill.objects.filter(pk=other.id).exists()


# ---- update / delete ------------------------------------------------------


@pytest.mark.django_db
def test_update_skill_partial(alice_client, alice):
    skill = Skill.objects.create(user=alice, name="X", body="old")
    r = alice_client.put(
        f"/api/skills/{skill.id}/",
        data=json.dumps({"body": "new"}),
        content_type="application/json",
    )
    assert r.status_code == 200, r.content
    skill.refresh_from_db()
    assert skill.body == "new"
    assert skill.name == "X"  # unchanged


@pytest.mark.django_db
def test_toggle_enabled(alice_client, alice):
    skill = Skill.objects.create(user=alice, name="X", body="x", enabled=True)
    r = alice_client.put(
        f"/api/skills/{skill.id}/",
        data=json.dumps({"enabled": False}),
        content_type="application/json",
    )
    assert r.status_code == 200
    skill.refresh_from_db()
    assert skill.enabled is False


@pytest.mark.django_db
def test_delete_skill(alice_client, alice):
    skill = Skill.objects.create(user=alice, name="X", body="x")
    r = alice_client.delete(f"/api/skills/{skill.id}/")
    assert r.status_code == 200
    assert not Skill.objects.filter(pk=skill.id).exists()
