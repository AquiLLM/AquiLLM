"""Crash windows retain either durable intent or an atomic committed result."""
from datetime import timedelta
import pytest
from django.utils import timezone
from apps.documents.models import RawTextDocument
from apps.ingestion.models import IngestionBatchItem
from aquillm.task_ingest_uploaded import run_ingest_uploaded_file
from .output_support import upload

pytestmark = pytest.mark.django_db(transaction=True)


def test_crash_after_blob_persistence_restarts_manifest(upload, monkeypatch):
    from apps.ingestion.services import source_artifacts
    item = upload((('one.txt', b'body'),))
    original = source_artifacts._put_if_absent
    def crash(*args, **kwargs):
        original(*args, **kwargs)
        raise SystemExit('after blob')
    monkeypatch.setattr(source_artifacts, '_put_if_absent', crash)
    with pytest.raises(SystemExit):
        run_ingest_uploaded_file(item.pk)
    assert not RawTextDocument.objects.exists()
    monkeypatch.setattr(source_artifacts, '_put_if_absent', original)
    IngestionBatchItem.objects.filter(pk=item.pk).update(next_attempt_at=timezone.now()-timedelta(seconds=1))
    run_ingest_uploaded_file(item.pk)
    item.refresh_from_db()
    assert item.document_ids == [str(RawTextDocument.objects.get().id)]


def test_crash_after_checkpoint_does_not_duplicate_output(upload, monkeypatch):
    from aquillm import task_ingest_uploaded as task
    item = upload()
    original = task.publish_output
    def crash(*args, **kwargs):
        result = original(*args, **kwargs)
        assert result.status == 'success'
        raise SystemExit('after checkpoint')
    monkeypatch.setattr(task, 'publish_output', crash)
    with pytest.raises(SystemExit):
        task.run_ingest_uploaded_file(item.pk)
    item.refresh_from_db()
    first = item.document_ids[0]
    assert first == str(RawTextDocument.objects.get().id)
    monkeypatch.setattr(task, 'publish_output', original)
    task.run_ingest_uploaded_file(item.pk)
    item.refresh_from_db()
    assert item.document_ids[0] == first
    assert RawTextDocument.objects.count() == 2


def test_crash_after_broker_ack_recovers_same_output(upload, monkeypatch):
    from apps.ingestion.models import IngestionOutput
    from apps.ingestion.services.output_manifest import prepare_output_manifest
    from apps.ingestion.services.output_dispatch import recover_due_outputs
    from apps.ingestion.tasks.output_recovery import publish_ingestion_output
    item = upload((('one.txt', b'body'),))
    prepare_output_manifest(item.pk, using='default')
    deliveries = []
    def ack(*args, **kwargs):
        deliveries.append((args, kwargs))
        raise SystemExit('ack then worker loss')
    monkeypatch.setattr(publish_ingestion_output, 'delay', ack)
    with pytest.raises(SystemExit):
        recover_due_outputs(limit=25, using='default')
    IngestionOutput.objects.update(next_attempt_at=timezone.now()-timedelta(seconds=1))
    monkeypatch.setattr(publish_ingestion_output, 'delay', publish_ingestion_output.run)
    assert recover_due_outputs(limit=25, using='default') == 1
    args, kwargs = deliveries[0]
    publish_ingestion_output.run(*args, **kwargs)
    item.refresh_from_db()
    assert item.document_ids == [str(RawTextDocument.objects.get().id)]
