"""Durable dispatch remains due when the broker is unavailable."""
import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from apps.ingestion.services.upload_batches import enqueue_upload_batch_files
from .output_support import upload

pytestmark = pytest.mark.django_db(transaction=True)


def test_broker_failure_preserves_accepted_item(upload, monkeypatch):
    item = upload()
    def fail(*args, **kwargs):
        raise ConnectionError('broker unavailable')
    monkeypatch.setattr('aquillm.tasks.ingest_uploaded_file_task.delay', fail)
    body, status = enqueue_upload_batch_files(item.batch.user, item.batch.collection,
        [SimpleUploadedFile('accepted.txt', b'accepted')], max_files=10, max_file_bytes=100)
    assert status == 202
    assert body['queued_count'] == 1


def test_task_registry_and_gated_schedule_preserve_existing_workers():
    from aquillm.celery import app
    from aquillm.celery_schedules import application_maintenance_schedule
    app.loader.import_default_modules()
    entries = application_maintenance_schedule(enabled=True)
    for name in ('document-chunk-publication-recovery', 'conversation-memory-recovery',
                 'profile-memory-reconciliation-recovery', 'ingestion-output-recovery'):
        assert entries[name]['task'] in app.tasks
    assert 'documents.build_embedding_generation' in app.tasks
    assert application_maintenance_schedule(enabled=False) == {}


def test_recovery_claims_only_requested_batch_and_leases_broker_ack(upload, monkeypatch):
    from apps.ingestion.models import IngestionOutput
    from apps.ingestion.services.output_manifest import prepare_output_manifest
    from apps.ingestion.services.output_dispatch import recover_due_outputs
    monkeypatch.setattr('apps.ingestion.tasks.output_recovery.publish_ingestion_output.delay', lambda *a, **k: None)
    item = upload(tuple((f'{i}.txt', f'body {i}'.encode()) for i in range(30)))
    prepare_output_manifest(item.pk, using='default')
    assert recover_due_outputs(limit=25, using='default') == 25
    assert IngestionOutput.objects.filter(status='processing').count() == 25
    assert recover_due_outputs(limit=25, using='default') == 5
    assert recover_due_outputs(limit=25, using='default') == 0


def test_repeated_worker_loss_is_bounded_and_terminal(upload):
    from datetime import timedelta
    from django.utils import timezone
    from apps.ingestion.models import IngestionOutput
    from apps.ingestion.services.output_manifest import prepare_output_manifest
    from apps.ingestion.services.output_ledger import claim_output
    item = upload((('one.txt', b'body'),))
    key, = prepare_output_manifest(item.pk, using='default')
    for _ in range(5):
        assert claim_output(key, using='default') is not None
        IngestionOutput.objects.update(next_attempt_at=timezone.now()-timedelta(seconds=1))
    assert claim_output(key, using='default') is None
    row = IngestionOutput.objects.get()
    assert row.status == 'error' and row.error_code == 'retry_exhausted'


def test_upload_blob_is_stored_before_dispatch_transaction(upload, monkeypatch):
    from django.core.files.storage import default_storage
    from django.db import connection
    item = upload()
    original = default_storage.save
    def save(*args, **kwargs):
        assert not connection.in_atomic_block
        return original(*args, **kwargs)
    monkeypatch.setattr(default_storage, 'save', save)
    monkeypatch.setattr('aquillm.tasks.ingest_uploaded_file_task.delay', lambda *a, **k: None)
    body, status = enqueue_upload_batch_files(item.batch.user, item.batch.collection,
        [SimpleUploadedFile('accepted.txt', b'accepted')], max_files=10, max_file_bytes=100)
    assert status == 202 and body['queued_count'] == 1


def test_transient_materialization_failure_retries_same_manifest(upload, monkeypatch):
    from datetime import timedelta
    from django.utils import timezone
    from apps.ingestion.models import IngestionOutput
    from apps.ingestion.services.output_manifest import prepare_output_manifest
    from apps.ingestion.services import output_ledger as ledger
    item = upload((('one.txt', b'body'),))
    key, = prepare_output_manifest(item.pk, using='default')
    original = ledger.verify_materialized
    def unavailable(_):
        raise TimeoutError('private storage endpoint')
    monkeypatch.setattr(ledger, 'verify_materialized', unavailable)
    first = ledger.publish_output(ledger.claim_output(key, using='default'), using='default')
    assert first.status == 'queued' and first.error_code == 'upstream_unavailable'
    assert ledger.claim_output(key, using='default') is None
    IngestionOutput.objects.update(next_attempt_at=timezone.now()-timedelta(seconds=1))
    monkeypatch.setattr(ledger, 'verify_materialized', original)
    second = ledger.publish_output(ledger.claim_output(key, using='default'), using='default')
    assert second.status == 'success' and second.output_key == first.output_key
    assert IngestionOutput.objects.count() == 1


def test_expired_parser_recovery_fences_generation_and_bounds_crashes(upload, monkeypatch):
    from datetime import timedelta
    from django.utils import timezone
    from apps.ingestion.models import IngestionOutput
    from apps.ingestion.services.output_manifest import prepare_output_manifest
    from apps.ingestion.services.output_dispatch import dispatch_item
    item = upload()
    def crash(*args, **kwargs):
        raise SystemExit('parser worker lost')
    monkeypatch.setattr('aquillm.ingestion.parsers.extract_text_payloads', crash)
    monkeypatch.setattr('aquillm.tasks.ingest_uploaded_file_task.delay', lambda *a, **k: None)
    for attempt in range(1, 6):
        with pytest.raises(SystemExit):
            prepare_output_manifest(item.pk, using='default')
        item.refresh_from_db()
        previous = item.generation
        assert item.attempts == attempt
        item.next_attempt_at = timezone.now()-timedelta(seconds=1)
        item.save(update_fields=['next_attempt_at'])
        assert dispatch_item(item.pk)
        item.refresh_from_db()
        assert item.generation != previous and item.attempts == attempt
    prepare_output_manifest(item.pk, using='default')
    item.refresh_from_db()
    assert item.status == 'error' and item.manifest_complete
    assert IngestionOutput.objects.get().error_code == 'retry_exhausted'
