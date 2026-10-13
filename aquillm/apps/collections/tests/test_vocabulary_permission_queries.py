"""Each vocabulary permission boundary reads fresh, bounded database authority."""

import pytest
from django.contrib.auth import get_user_model
from django.db import connection, transaction
from django.test.utils import CaptureQueriesContext

from apps.collections.models import Collection, CollectionPermission
from apps.collections.services.vocabulary_access import require_permission

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.fixture
def actor():
    return get_user_model().objects.create_user(username="vocabulary-permission-reader")


def chain(length):
    return Collection.objects.bulk_create([
        Collection(id=41000 + number, name=f"permission-level-{number}",
                   parent_id=41000 + number - 1 if number > 1 else None)
        for number in range(1, length + 1)
    ])


def set_parent(collection_id, parent_id):
    # Corrupt hierarchy fixtures must reach the real reader, without ORM guards.
    with connection.cursor() as cursor:
        cursor.execute("UPDATE aquillm_collection SET parent_id = %s WHERE id = %s",
                       [parent_id, collection_id])


@pytest.mark.parametrize("depth,expected", [
    (1, (41001,)),
    (32, (41032, 41031, 41030, 41029, 41028, 41027, 41026, 41025,
          41024, 41023, 41022, 41021, 41020, 41019, 41018, 41017,
          41016, 41015, 41014, 41013, 41012, 41011, 41010, 41009,
          41008, 41007, 41006, 41005, 41004, 41003, 41002, 41001)),
])
def test_permission_boundary_uses_one_statement_at_root_and_depth_limit(actor, depth, expected):
    nodes = chain(depth)
    CollectionPermission.objects.create(user=actor, collection=nodes[0], permission="VIEW")

    with CaptureQueriesContext(connection) as queries:
        path = require_permission(actor, nodes[-1].pk, "default")

    assert path == expected
    assert 0 < len(queries) <= 1


@pytest.mark.parametrize("granted,requested", [
    ("VIEW", "VIEW"), ("EDIT", "VIEW"), ("MANAGE", "VIEW"), ("MANAGE", "MANAGE"),
])
def test_ancestor_permission_authorizes_the_exact_descendant_path(actor, granted, requested):
    root, child = chain(2)
    CollectionPermission.objects.create(user=actor, collection=root, permission=granted)

    assert require_permission(actor, child.pk, "default", requested) == (41002, 41001)


@pytest.mark.parametrize("granted", ["VIEW", "EDIT"])
def test_manage_does_not_accept_lower_ancestor_permission(actor, granted):
    root, child = chain(2)
    CollectionPermission.objects.create(user=actor, collection=root, permission=granted)

    with pytest.raises(PermissionError):
        require_permission(actor, child.pk, "default", "MANAGE")


def test_persisted_inactive_actor_is_rejected_despite_active_python_instance(actor):
    root, = chain(1)
    CollectionPermission.objects.create(user=actor, collection=root, permission="MANAGE")
    get_user_model().objects.filter(pk=actor.pk).update(is_active=False)
    assert actor.is_active is True

    with pytest.raises(PermissionError):
        require_permission(actor, root.pk, "default")


def test_repeated_boundary_observes_revoked_permission(actor):
    root, child = chain(2)
    permission = CollectionPermission.objects.create(user=actor, collection=root, permission="VIEW")
    assert require_permission(actor, child.pk, "default") == (41002, 41001)

    permission.delete()

    with pytest.raises(PermissionError):
        require_permission(actor, child.pk, "default")


def test_permission_on_unrelated_collection_does_not_authorize_selected_path(actor):
    root, child = chain(2)
    unrelated = Collection.objects.create(name="unrelated permission")
    CollectionPermission.objects.create(user=actor, collection=unrelated, permission="MANAGE")

    with pytest.raises(PermissionError):
        require_permission(actor, child.pk, "default")


def test_missing_collection_is_rejected(actor):
    root, = chain(1)
    missing_id = root.pk
    root.delete()

    with pytest.raises(ValueError):
        require_permission(actor, missing_id, "default")


def test_missing_ancestor_is_rejected_before_own_permission_can_authorize(actor):
    root, = chain(1)
    CollectionPermission.objects.create(user=actor, collection=root, permission="MANAGE")
    with transaction.atomic():
        # PostgreSQL's deferred FK permits a transient dangling edge; restore it
        # before commit instead of disabling database integrity checks.
        set_parent(root.pk, 41999)
        try:
            with pytest.raises(ValueError):
                require_permission(actor, root.pk, "default")
        finally:
            set_parent(root.pk, None)


@pytest.mark.parametrize("depth", [1, 2])
def test_cycle_is_rejected_even_with_permission_on_the_cycle(actor, depth):
    nodes = chain(depth)
    CollectionPermission.objects.create(user=actor, collection=nodes[0], permission="MANAGE")
    set_parent(nodes[0].pk, nodes[-1].pk)

    with pytest.raises(ValueError):
        require_permission(actor, nodes[-1].pk, "default")


def test_thirty_third_ancestor_cannot_be_silently_truncated(actor):
    nodes = chain(33)
    CollectionPermission.objects.create(user=actor, collection=nodes[-1], permission="MANAGE")

    with pytest.raises(ValueError):
        require_permission(actor, nodes[-1].pk, "default")


def test_none_collection_is_permission_denied(actor):
    root, = chain(1)
    CollectionPermission.objects.create(user=actor, collection=root, permission="MANAGE")

    with pytest.raises(PermissionError):
        require_permission(actor, None, "default")
