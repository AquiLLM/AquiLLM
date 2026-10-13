"""Original positions survive atomic acceptance of same-name uploads."""
import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from apps.ingestion.models import IngestionBatchItem
from apps.ingestion.services.upload_batches import enqueue_upload_batch_files
from .output_support import upload


@pytest.mark.django_db(transaction=True)
def test_acceptance_and_rejections_identify_original_file_position(upload, monkeypatch):
    seed = upload()
    monkeypatch.setattr('aquillm.tasks.ingest_uploaded_file_task.delay', lambda *a, **k: None)
    files = [SimpleUploadedFile('same.txt', body) for body in (b'ok', b'', b'oversized', b'yes')]
    payload, status = enqueue_upload_batch_files(seed.batch.user, seed.batch.collection,
        files, max_files=10, max_file_bytes=5)
    assert status == 202
    assert [item['source_index'] for item in payload['items']] == [0, 3]
    assert [item['source_index'] for item in payload['rejected']] == [1, 2]
    accepted = IngestionBatchItem.objects.filter(pk__in=[item['id'] for item in payload['items']])
    assert accepted.count() == 2
    assert len(set(accepted.values_list('source_file', flat=True))) == 2
