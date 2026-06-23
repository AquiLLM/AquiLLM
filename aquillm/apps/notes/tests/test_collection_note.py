"""Tests for per-collection Collection Notes: loader + API permission checks."""
import json

import pytest
from django.contrib.auth import get_user_model
from django.test import Client

from apps.collections.models import Collection, CollectionPermission
from apps.notes.models import CollectionNote
from apps.notes.services.runtime import (
    aload_collection_note_bodies,
    load_collection_note_bodies,
)

User = get_user_model()


# ---- fixtures --------------------------------------------------------------


@pytest.fixture
def owner(db):
    return User.objects.create_user(username="owner", password="pw")


@pytest.fixture
def editor(db):
    return User.objects.create_user(username="editor", password="pw")


@pytest.fixture
def viewer(db):
    return User.objects.create_user(username="viewer", password="pw")


@pytest.fixture
def stranger(db):
    return User.objects.create_user(username="stranger", password="pw")


@pytest.fixture
def collection(owner, editor, viewer):
    c = Collection.objects.create(name="papers")
    CollectionPermission.objects.create(user=owner, collection=c, permission="MANAGE")
    CollectionPermission.objects.create(user=editor, collection=c, permission="EDIT")
    CollectionPermission.objects.create(user=viewer, collection=c, permission="VIEW")
    return c


def _client(user):
    c = Client()
    c.force_login(user)
    return c


# ---- loader ----------------------------------------------------------------


@pytest.mark.django_db
def test_loader_empty_when_no_row(collection):
    assert load_collection_note_bodies([collection.id]) == ""


@pytest.mark.django_db
def test_loader_empty_when_collection_list_empty():
    assert load_collection_note_bodies([]) == ""


@pytest.mark.django_db
def test_loader_includes_collection_name_header(collection):
    CollectionNote.objects.create(collection=collection, body="The authors are X and Y.")
    out = load_collection_note_bodies([collection.id])
    assert "## Collection notes — papers" in out
    assert "The authors are X and Y." in out


@pytest.mark.django_db
def test_loader_concatenates_multiple_collections(owner):
    c1 = Collection.objects.create(name="alpha-coll")
    c2 = Collection.objects.create(name="beta-coll")
    CollectionNote.objects.create(collection=c1, body="alpha body")
    CollectionNote.objects.create(collection=c2, body="beta body")
    out = load_collection_note_bodies([c1.id, c2.id])
    assert "alpha body" in out
    assert "beta body" in out
    assert "\n\n---\n\n" in out  # section separator


@pytest.mark.django_db
def test_loader_skips_empty_bodies(collection):
    CollectionNote.objects.create(collection=collection, body="   ")
    assert load_collection_note_bodies([collection.id]) == ""


@pytest.mark.django_db(transaction=True)
async def test_async_loader_works(collection):
    await CollectionNote.objects.acreate(collection=collection, body="async body")
    out = await aload_collection_note_bodies([collection.id])
    assert "async body" in out


# ---- API: permission gates -------------------------------------------------


@pytest.mark.django_db
def test_get_returns_empty_shape_when_no_row(collection, owner):
    r = _client(owner).get(f"/api/collections/{collection.id}/note/")
    assert r.status_code == 200
    data = r.json()
    assert data["body"] == ""
    assert data["exists"] is False
    assert data["max_body_length"] > 0


@pytest.mark.django_db
def test_get_allowed_for_editor(collection, editor):
    r = _client(editor).get(f"/api/collections/{collection.id}/note/")
    assert r.status_code == 200


@pytest.mark.django_db
def test_get_forbidden_for_viewer(collection, viewer):
    """VIEW alone is not enough to see the notes text (conservative default)."""
    r = _client(viewer).get(f"/api/collections/{collection.id}/note/")
    assert r.status_code == 403


@pytest.mark.django_db
def test_get_forbidden_for_stranger(collection, stranger):
    r = _client(stranger).get(f"/api/collections/{collection.id}/note/")
    assert r.status_code == 403


@pytest.mark.django_db
def test_put_upserts_for_manager(collection, owner):
    r = _client(owner).put(
        f"/api/collections/{collection.id}/note/",
        data=json.dumps({"body": "first revision"}),
        content_type="application/json",
    )
    assert r.status_code == 200
    assert CollectionNote.objects.get(collection=collection).body == "first revision"

    # second PUT should update, not duplicate
    r2 = _client(owner).put(
        f"/api/collections/{collection.id}/note/",
        data=json.dumps({"body": "second revision"}),
        content_type="application/json",
    )
    assert r2.status_code == 200
    assert CollectionNote.objects.filter(collection=collection).count() == 1
    assert CollectionNote.objects.get(collection=collection).body == "second revision"


@pytest.mark.django_db
def test_put_forbidden_for_editor(collection, editor):
    r = _client(editor).put(
        f"/api/collections/{collection.id}/note/",
        data=json.dumps({"body": "should not save"}),
        content_type="application/json",
    )
    assert r.status_code == 403
    assert not CollectionNote.objects.filter(collection=collection).exists()


@pytest.mark.django_db
def test_put_rejects_oversized_body(collection, owner):
    big = "x" * 17_000  # over the 16 KB cap
    r = _client(owner).put(
        f"/api/collections/{collection.id}/note/",
        data=json.dumps({"body": big}),
        content_type="application/json",
    )
    assert r.status_code == 400
    assert not CollectionNote.objects.filter(collection=collection).exists()


@pytest.mark.django_db
def test_put_records_updated_by(collection, owner):
    _client(owner).put(
        f"/api/collections/{collection.id}/note/",
        data=json.dumps({"body": "by-owner"}),
        content_type="application/json",
    )
    note = CollectionNote.objects.get(collection=collection)
    assert note.updated_by_id == owner.id
