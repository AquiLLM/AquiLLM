"""PostgreSQL bypass negatives and stale parser recovery without fence reuse."""
from datetime import timedelta
import pytest
from django.db import DatabaseError, connection, transaction
from django.utils import timezone
from apps.ingestion.models import IngestionBatchItem
from apps.ingestion.services.output_manifest import prepare_output_manifest
from apps.ingestion.services.output_dispatch import dispatch_item
from .output_support import upload

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.mark.parametrize('writer', ['bulk', 'raw'])
@pytest.mark.parametrize('completed', [False, True])
def test_attempt_zero_cannot_unlock_claimed_source(upload, monkeypatch, writer, completed):
    item = upload()
    if completed:
        prepare_output_manifest(item.pk, using='default')
    else:
        def crash(*args, **kwargs):
            raise SystemExit('parser lost')
        monkeypatch.setattr('aquillm.ingestion.parsers.extract_text_payloads', crash)
        with pytest.raises(SystemExit):
            prepare_output_manifest(item.pk, using='default')
    with pytest.raises(DatabaseError), transaction.atomic():
        if writer == 'bulk':
            IngestionBatchItem.objects.filter(pk=item.pk).update(attempts=0)
            IngestionBatchItem.objects.filter(pk=item.pk).update(source_file='forged.txt')
        else:
            with connection.cursor() as cursor:
                cursor.execute('UPDATE aquillm_ingestionbatchitem SET attempts=0 WHERE id=%s', [item.pk])
                cursor.execute('UPDATE aquillm_ingestionbatchitem SET original_filename=%s WHERE id=%s', ['forged.txt', item.pk])
    item.refresh_from_db()
    assert item.attempts == 1 and item.original_filename == 'sources.zip'


def test_completed_manifest_freezes_source_even_without_attempt_counter(upload):
    item = upload()
    IngestionBatchItem.objects.filter(pk=item.pk).update(manifest_complete=True)
    with pytest.raises(DatabaseError), transaction.atomic():
        IngestionBatchItem.objects.filter(pk=item.pk).update(content_type='forged/type')


def test_recovery_generation_reset_rejects_stale_parser_completion(upload, monkeypatch):
    from aquillm.ingestion import parsers
    original = parsers.extract_text_payloads
    item = upload((('one.txt', b'body'),))
    monkeypatch.setattr('aquillm.tasks.ingest_uploaded_file_task.delay', lambda *a, **k: None)
    def reset_during_parse(*args, **kwargs):
        result = original(*args, **kwargs)
        if str(args[0]).endswith('.zip'):
            IngestionBatchItem.objects.filter(pk=item.pk).update(next_attempt_at=timezone.now()-timedelta(seconds=1))
            assert dispatch_item(item.pk)
        return result
    monkeypatch.setattr(parsers, 'extract_text_payloads', reset_during_parse)
    assert prepare_output_manifest(item.pk, using='default') == ()
    item.refresh_from_db()
    assert item.attempts == 1 and not item.manifest_complete and not item.outputs.exists()
    monkeypatch.setattr(parsers, 'extract_text_payloads', original)
    assert len(prepare_output_manifest(item.pk, using='default')) == 1
    item.refresh_from_db()
    assert item.attempts == 2 and item.manifest_complete


def test_processing_claim_cannot_reuse_attempt_pair(upload):
    item = upload()
    with pytest.raises(DatabaseError), transaction.atomic():
        IngestionBatchItem.objects.filter(pk=item.pk).update(status='processing')
