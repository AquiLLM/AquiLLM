import json

import pytest
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse

from aquillm.models import (
    Collection,
    CollectionPermission,
    DocumentFigure,
    IngestionBatch,
    IngestionBatchItem,
    RawTextDocument,
)


@pytest.mark.django_db
def test_collection_detail_uses_parser_type_for_raw_text_documents(client, settings, tmp_path):
    settings.STORAGES = {
        "default": {
            "BACKEND": "django.core.files.storage.FileSystemStorage",
            "OPTIONS": {"location": str(tmp_path)},
        },
        "staticfiles": {"BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"},
    }

    user = User.objects.create_user(username="collection-api-user", password="pw12345")
    collection = Collection.objects.create(name="Slides Collection")
    CollectionPermission.objects.create(user=user, collection=collection, permission="VIEW")

    doc = RawTextDocument.objects.create(
        title="Lecture1",
        full_text="Slide deck text",
        collection=collection,
        ingested_by=user,
    )

    batch = IngestionBatch.objects.create(user=user, collection=collection)
    IngestionBatchItem.objects.create(
        batch=batch,
        source_file=SimpleUploadedFile(
            "lecture1.pptx",
            b"pptx",
            content_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ),
        original_filename="lecture1.pptx",
        content_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        file_size_bytes=4,
        status=IngestionBatchItem.Status.SUCCESS,
        document_ids=[str(doc.id)],
        parser_metadata={
            "outputs": [
                {
                    "document_id": str(doc.id),
                    "document_model": "RawTextDocument",
                    "normalized_type": "pptx",
                    "modality": "text",
                }
            ]
        },
    )

    assert client.login(username="collection-api-user", password="pw12345")
    response = client.get(reverse("api_collection", kwargs={"col_id": collection.id}))

    assert response.status_code == 200
    payload = response.json()
    listed_doc = next(item for item in payload["documents"] if item["id"] == str(doc.id))
    assert listed_doc["type"] == "PPTX"
    assert listed_doc["model_type"] == "RawTextDocument"


@pytest.mark.django_db
def test_collection_detail_includes_parent_document_id_for_figure_subcollection(client, settings, tmp_path):
    settings.STORAGES = {
        "default": {
            "BACKEND": "django.core.files.storage.FileSystemStorage",
            "OPTIONS": {"location": str(tmp_path)},
        },
        "staticfiles": {"BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"},
    }

    user = User.objects.create_user(username="collection-figure-user", password="pw12345")
    collection = Collection.objects.create(name="Root Collection")
    CollectionPermission.objects.create(user=user, collection=collection, permission="VIEW")
    CollectionPermission.objects.create(user=user, collection=collection, permission="EDIT")

    source_doc = RawTextDocument.objects.create(
        title="Lecture1",
        full_text="slides content",
        collection=collection,
        ingested_by=user,
    )
    figures_collection = Collection.objects.create(name="Lecture1 - Figures", parent=collection)
    CollectionPermission.objects.create(user=user, collection=figures_collection, permission="VIEW")

    figure = DocumentFigure(
        title="Lecture1 - Figure 1",
        full_text="figure text",
        collection=figures_collection,
        ingested_by=user,
        source_format="pptx",
        figure_index=0,
        extracted_caption="caption",
        location_metadata={"slide_number": 1},
    )
    figure.parent_document = source_doc
    figure.image_file = SimpleUploadedFile("figure1.png", b"\x89PNG\r\n\x1a\n")
    figure.save()

    assert client.login(username="collection-figure-user", password="pw12345")
    response = client.get(reverse("api_collection", kwargs={"col_id": collection.id}))

    assert response.status_code == 200
    payload = response.json()
    listed_child = next(item for item in payload["children"] if item["id"] == figures_collection.id)
    assert listed_child["parent_document_id"] == str(source_doc.id)


@pytest.mark.django_db
def test_collection_detail_infers_parent_document_id_from_figure_title_when_link_missing(client, settings, tmp_path):
    settings.STORAGES = {
        "default": {
            "BACKEND": "django.core.files.storage.FileSystemStorage",
            "OPTIONS": {"location": str(tmp_path)},
        },
        "staticfiles": {"BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"},
    }

    user = User.objects.create_user(username="collection-figure-title-user", password="pw12345")
    collection = Collection.objects.create(name="Root Collection")
    CollectionPermission.objects.create(user=user, collection=collection, permission="VIEW")
    CollectionPermission.objects.create(user=user, collection=collection, permission="EDIT")

    source_doc = RawTextDocument.objects.create(
        title="Lecture1",
        full_text="slides content",
        collection=collection,
        ingested_by=user,
    )
    figures_collection = Collection.objects.create(name="Lecture1 - Figures", parent=collection)
    CollectionPermission.objects.create(user=user, collection=figures_collection, permission="VIEW")

    figure = DocumentFigure(
        title="Lecture1 - Figure 1",
        full_text="figure text",
        collection=figures_collection,
        ingested_by=user,
        source_format="pptx",
        figure_index=0,
        extracted_caption="caption",
        location_metadata={"slide_number": 1},
    )
    figure.image_file = SimpleUploadedFile("figure1.png", b"\x89PNG\r\n\x1a\n")
    figure.save()

    assert client.login(username="collection-figure-title-user", password="pw12345")
    response = client.get(reverse("api_collection", kwargs={"col_id": collection.id}))

    assert response.status_code == 200
    payload = response.json()
    listed_child = next(item for item in payload["children"] if item["id"] == figures_collection.id)
    assert listed_child["parent_document_id"] == str(source_doc.id)


@pytest.mark.django_db
def test_collection_permissions_post_syncs_figure_subcollections(client):
    owner = User.objects.create_user(username="perm-owner", password="pw12345")
    collaborator = User.objects.create_user(username="perm-collab", password="pw12345")
    collection = Collection.objects.create(name="Shared Root")
    CollectionPermission.objects.create(user=owner, collection=collection, permission="MANAGE")
    figures_collection = Collection.objects.create(name="Lecture1 - Figures", parent=collection)
    CollectionPermission.objects.create(user=owner, collection=figures_collection, permission="MANAGE")

    assert client.login(username="perm-owner", password="pw12345")
    url = reverse("api_collection_permissions", kwargs={"col_id": collection.id})
    response = client.post(
        url,
        data=json.dumps({"viewers": [collaborator.id], "editors": [], "admins": []}),
        content_type="application/json",
    )
    assert response.status_code == 200

    fig_view = CollectionPermission.objects.filter(
        collection=figures_collection, user=collaborator, permission="VIEW"
    ).exists()
    assert fig_view is True


@pytest.mark.django_db
def test_collection_permissions_post_removes_collaborator_from_figure_subcollections(client):
    owner = User.objects.create_user(username="perm-owner2", password="pw12345")
    collaborator = User.objects.create_user(username="perm-collab2", password="pw12345")
    collection = Collection.objects.create(name="Shared Root 2")
    CollectionPermission.objects.create(user=owner, collection=collection, permission="MANAGE")
    figures_collection = Collection.objects.create(name="Doc - Figures", parent=collection)
    CollectionPermission.objects.create(user=owner, collection=figures_collection, permission="MANAGE")
    CollectionPermission.objects.create(user=collaborator, collection=collection, permission="VIEW")
    CollectionPermission.objects.create(user=collaborator, collection=figures_collection, permission="VIEW")

    assert client.login(username="perm-owner2", password="pw12345")
    url = reverse("api_collection_permissions", kwargs={"col_id": collection.id})
    response = client.post(
        url,
        data=json.dumps({"viewers": [], "editors": [], "admins": []}),
        content_type="application/json",
    )
    assert response.status_code == 200

    assert not CollectionPermission.objects.filter(collection=figures_collection, user=collaborator).exists()


@pytest.mark.django_db
def test_collection_permissions_post_leaves_non_figure_children_untouched(client):
    owner = User.objects.create_user(username="perm-owner3", password="pw12345")
    collection = Collection.objects.create(name="Shared Root 3")
    CollectionPermission.objects.create(user=owner, collection=collection, permission="MANAGE")
    other_child = Collection.objects.create(name="My Notes", parent=collection)
    CollectionPermission.objects.create(user=owner, collection=other_child, permission="MANAGE")

    assert client.login(username="perm-owner3", password="pw12345")
    url = reverse("api_collection_permissions", kwargs={"col_id": collection.id})
    response = client.post(
        url,
        data=json.dumps({"viewers": [], "editors": [], "admins": []}),
        content_type="application/json",
    )
    assert response.status_code == 200

    assert CollectionPermission.objects.filter(collection=other_child, user=owner, permission="MANAGE").exists()
