import pytest
from django.contrib.auth.models import User
from aquillm.models import Collection, CollectionPermission

@pytest.mark.django_db
def test_collection_creation():
    collection = Collection.objects.create(name="Test Collection")
    assert collection.name == "Test Collection"
    assert collection.pk is not None

@pytest.mark.django_db
def test_collection_str():
    collection = Collection.objects.create(name="Test Collection")
    assert str(collection) == "Test Collection"

@pytest.mark.django_db
def test_collection_user_permissions():
    user = User.objects.create_user(username='testuser', password='12345')
    collection = Collection.objects.create(name="Test Collection")
    CollectionPermission.objects.create(user=user, collection=collection, permission='VIEW')

    assert collection.user_can_view(user) is True
    assert collection.user_can_edit(user) is False
    assert collection.user_can_manage(user) is False

@pytest.mark.django_db
def test_collection_user_permissions_edit():
    user = User.objects.create_user(username='testuser', password='12345')
    collection = Collection.objects.create(name="Test Collection")
    CollectionPermission.objects.create(user=user, collection=collection, permission='EDIT')

    assert collection.user_can_view(user) is True
    assert collection.user_can_edit(user) is True
    assert collection.user_can_manage(user) is False

@pytest.mark.django_db
def test_collection_user_permissions_manage():
    user = User.objects.create_user(username='testuser', password='12345')
    collection = Collection.objects.create(name="Test Collection")
    CollectionPermission.objects.create(user=user, collection=collection, permission='MANAGE')

    assert collection.user_can_view(user) is True
    assert collection.user_can_edit(user) is True
    assert collection.user_can_manage(user) is True

@pytest.mark.django_db
def test_collection_get_user_accessible_documents():
    user = User.objects.create_user(username='testuser', password='12345')
    collection1 = Collection.objects.create(name="Collection 1")
    collection2 = Collection.objects.create(name="Collection 2")
    CollectionPermission.objects.create(user=user, collection=collection1, permission='VIEW')

    accessible_docs = Collection.get_user_accessible_documents(user)
    assert len(accessible_docs) == 0  # Assuming no documents are added yet

@pytest.mark.django_db
def test_collection_filter_by_user_perm():
    user = User.objects.create_user(username='testuser', password='12345')
    collection1 = Collection.objects.create(name="Collection 1")
    collection2 = Collection.objects.create(name="Collection 2")
    CollectionPermission.objects.create(user=user, collection=collection1, permission='VIEW')

    filtered_collections = Collection.objects.filter_by_user_perm(user, 'VIEW')
    assert len(filtered_collections) == 1
    assert filtered_collections[0] == collection1
    assert False