from unittest.mock import patch

import pytest
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse

from aquillm.models import Collection, CollectionPermission, IngestionBatch, IngestionBatchItem


def test_ingestion_batch_item_source_file_allows_long_storage_paths():
    # S3/Django appends suffixes when resolving collisions; keep enough headroom.
    assert IngestionBatchItem._meta.get_field("source_file").max_length == 500


@pytest.mark.django_db
@patch("aquillm.api_views.ingest_uploaded_file_task.delay")
def test_upload_endpoint_queues_each_file(delay_mock, settings, tmp_path):
    settings.STORAGES = {
        "default": {
            "BACKEND": "django.core.files.storage.FileSystemStorage",
            "OPTIONS": {"location": str(tmp_path)},
        },
        "staticfiles": {"BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"},
    }
    user = User.objects.create_user(username="upload-user", password="pw12345")
    collection = Collection.objects.create(name="Upload Collection")
    CollectionPermission.objects.create(user=user, collection=collection, permission="EDIT")

    from django.test import Client

    client = Client()
    assert client.login(username="upload-user", password="pw12345")

    file_a = SimpleUploadedFile("notes.txt", b"hello world", content_type="text/plain")
    file_b = SimpleUploadedFile("table.csv", b"a,b\n1,2\n", content_type="text/csv")

    response = client.post(
        reverse("api_ingest_uploads"),
        data={"collection": str(collection.id), "files": [file_a, file_b]},
    )

    assert response.status_code == 202
    payload = response.json()
    assert payload["queued_count"] == 2
    assert len(payload["items"]) == 2
    assert delay_mock.call_count == 2


@pytest.mark.django_db
def test_status_endpoint_returns_counts(settings, tmp_path):
    settings.STORAGES = {
        "default": {
            "BACKEND": "django.core.files.storage.FileSystemStorage",
            "OPTIONS": {"location": str(tmp_path)},
        },
        "staticfiles": {"BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"},
    }
    user = User.objects.create_user(username="status-user", password="pw12345")
    collection = Collection.objects.create(name="Status Collection")
    CollectionPermission.objects.create(user=user, collection=collection, permission="EDIT")
    batch = IngestionBatch.objects.create(user=user, collection=collection)
    IngestionBatchItem.objects.create(
        batch=batch,
        source_file=SimpleUploadedFile("ok.txt", b"hello", content_type="text/plain"),
        original_filename="ok.txt",
        status=IngestionBatchItem.Status.QUEUED,
    )

    from django.test import Client

    client = Client()
    assert client.login(username="status-user", password="pw12345")
    response = client.get(reverse("api_ingest_uploads_status", kwargs={"batch_id": batch.id}))
    assert response.status_code == 200
    payload = response.json()
    assert payload["counts"]["queued"] == 1
    assert len(payload["items"]) == 1


@pytest.mark.django_db
def test_status_endpoint_includes_modality_metadata(settings, tmp_path):
    settings.STORAGES = {
        "default": {
            "BACKEND": "django.core.files.storage.FileSystemStorage",
            "OPTIONS": {"location": str(tmp_path)},
        },
        "staticfiles": {"BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"},
    }
    user = User.objects.create_user(username="status-meta-user", password="pw12345")
    collection = Collection.objects.create(name="Status Metadata Collection")
    CollectionPermission.objects.create(user=user, collection=collection, permission="EDIT")
    batch = IngestionBatch.objects.create(user=user, collection=collection)
    item = IngestionBatchItem.objects.create(
        batch=batch,
        source_file=SimpleUploadedFile("image.png", b"\x89PNG...", content_type="image/png"),
        original_filename="image.png",
        status=IngestionBatchItem.Status.SUCCESS,
        parser_metadata={
            "modalities": ["image"],
            "outputs": [
                {
                    "modality": "image",
                    "normalized_type": "image_ocr",
                    "provider": "qwen",
                    "raw_media_saved": True,
                    "text_extracted": True,
                }
            ],
        },
    )

    from django.test import Client

    client = Client()
    assert client.login(username="status-meta-user", password="pw12345")
    response = client.get(reverse("api_ingest_uploads_status", kwargs={"batch_id": batch.id}))
    assert response.status_code == 200
    payload = response.json()
    listed = next(row for row in payload["items"] if row["id"] == item.id)
    assert listed["modalities"] == ["image"]
    assert listed["providers"] == ["qwen"]
    assert listed["raw_media_saved"] is True
    assert listed["text_extracted"] is True
