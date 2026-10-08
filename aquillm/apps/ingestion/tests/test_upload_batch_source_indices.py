"""Retry identity for same-name uploaded files, without database/provider calls."""
from types import SimpleNamespace
from unittest.mock import patch

from django.core.files.uploadedfile import SimpleUploadedFile

from apps.ingestion.services.upload_batches import enqueue_upload_batch_files


def test_acceptance_and_rejections_identify_original_file_position():
    files = [
        SimpleUploadedFile("same.txt", b"ok"),
        SimpleUploadedFile("same.txt", b""),
        SimpleUploadedFile("same.txt", b"oversized"),
        SimpleUploadedFile("same.txt", b"yes"),
    ]
    with (
        patch("apps.ingestion.services.upload_batches.IngestionBatch.objects.create", return_value=SimpleNamespace(id=17)),
        patch("apps.ingestion.services.upload_batches.IngestionBatchItem.objects.create", side_effect=[
            SimpleNamespace(id=1, original_filename="same.txt", status="queued"),
            SimpleNamespace(id=2, original_filename="same.txt", status="queued"),
        ]),
        patch("apps.ingestion.services.upload_batches.ingest_uploaded_file_task.delay"),
    ):
        payload, status = enqueue_upload_batch_files(object(), object(), files, max_files=10, max_file_bytes=5)

    assert status == 202
    assert [item["source_index"] for item in payload["items"]] == [0, 3]
    assert [item["source_index"] for item in payload["rejected"]] == [1, 2]
