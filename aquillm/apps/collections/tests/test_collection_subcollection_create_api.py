import json

import pytest
from django.contrib.auth import get_user_model
from django.urls import reverse

from apps.collections.models import Collection, CollectionPermission

User = get_user_model()


@pytest.mark.django_db
def test_collections_create_can_create_subcollection_with_parent_id(client):
    user = User.objects.create_user(username="collections-child-user", password="pw12345")
    parent = Collection.objects.create(name="Parent Collection")
    CollectionPermission.objects.create(user=user, collection=parent, permission="EDIT")
    assert client.login(username="collections-child-user", password="pw12345")

    response = client.post(
        reverse("api_collections"),
        data=json.dumps({"name": "skill_pack", "parent_id": parent.id}),
        content_type="application/json",
    )

    assert response.status_code == 200
    payload = response.json()
    created = Collection.objects.get(id=payload["id"])
    assert created.parent == parent
    assert payload["parent"] == parent.id
    assert payload["path"] == "Parent Collection/skill_pack"
    assert CollectionPermission.objects.filter(
        collection=created,
        user=user,
        permission="MANAGE",
    ).exists()


@pytest.mark.django_db
def test_collections_create_subcollection_requires_parent_edit_permission(client):
    user = User.objects.create_user(username="collections-child-denied", password="pw12345")
    parent = Collection.objects.create(name="Parent Collection")
    CollectionPermission.objects.create(user=user, collection=parent, permission="VIEW")
    assert client.login(username="collections-child-denied", password="pw12345")

    response = client.post(
        reverse("api_collections"),
        data=json.dumps({"name": "skill_pack", "parent_id": parent.id}),
        content_type="application/json",
    )

    assert response.status_code == 403
