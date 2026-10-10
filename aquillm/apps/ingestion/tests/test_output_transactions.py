"""Real PostgreSQL rollback, fenced leases, and competing deliveries."""
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
import pytest
from django.db import close_old_connections, connection
from django.utils import timezone
from apps.documents.models import RawTextDocument
from .output_support import upload

pytestmark = pytest.mark.django_db(transaction=True)


def test_crash_after_document_insert_rolls_back_and_recovers(upload, monkeypatch):
    from apps.ingestion.services.output_manifest import prepare_output_manifest
    from apps.ingestion.services.output_ledger import claim_output, publish_output
    from apps.ingestion.models import IngestionOutput
    item = upload((('first.txt', b'first'),))
    key, = prepare_output_manifest(item.pk, using='default')
    original = RawTextDocument.save
    def crash(document, *args, **kwargs):
        original(document, *args, **kwargs)
        raise SystemExit('worker died after insert')
    monkeypatch.setattr(RawTextDocument, 'save', crash)
    lease = claim_output(key, using='default')
    with pytest.raises(SystemExit):
        publish_output(lease, using='default')
    assert RawTextDocument.objects.count() == 0
    monkeypatch.setattr(RawTextDocument, 'save', original)
    IngestionOutput.objects.filter(output_key=key).update(next_attempt_at=timezone.now()-timedelta(seconds=1))
    result = publish_output(claim_output(key, using='default'), using='default')
    item.refresh_from_db()
    assert item.document_ids == [str(result.document_ref.id)]


def test_two_real_connections_only_one_delivery_claims_output(upload):
    from apps.ingestion.services.output_manifest import prepare_output_manifest
    from apps.ingestion.services.output_ledger import claim_output, publish_output
    item = upload((('first.txt', b'first'),))
    key, = prepare_output_manifest(item.pk, using='default')
    def run():
        close_old_connections()
        try:
            lease = claim_output(key, using='default')
            return publish_output(lease, using='default') if lease else None
        finally:
            close_old_connections()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: run(), range(2)))
    assert sum(r is not None for r in results) == 1
    assert RawTextDocument.objects.count() == 1
    item.refresh_from_db()
    assert item.document_ids == [str(RawTextDocument.objects.get().id)]


def test_stale_worker_cannot_fail_reset_generation(upload):
    from apps.ingestion.services.output_manifest import prepare_output_manifest
    from apps.ingestion.services.output_ledger import claim_output, fail_output, reset_output
    from apps.ingestion.models import IngestionOutput
    item = upload((('first.txt', b'first'),))
    key, = prepare_output_manifest(item.pk, using='default')
    old = claim_output(key, using='default')
    fail_output(old, ValueError('invalid'), using='default')
    row = IngestionOutput.objects.get(output_key=key)
    assert reset_output(key, generation=old.generation,
                        extraction_revision_id=row.extraction_revision_id, using='default')
    new = claim_output(key, using='default')
    fail_output(old, ValueError('late private text'), using='default')
    row.refresh_from_db()
    assert row.generation == new.generation and row.status == 'processing'
    assert row.error_code == ''


def test_parser_and_blob_work_are_outside_transaction(upload, monkeypatch):
    from aquillm.ingestion import parsers
    from apps.ingestion.services.output_manifest import prepare_output_manifest
    item = upload()
    original = parsers.extract_text_payloads
    def extract(*args, **kwargs):
        assert not connection.in_atomic_block
        return original(*args, **kwargs)
    monkeypatch.setattr(parsers, 'extract_text_payloads', extract)
    assert len(prepare_output_manifest(item.pk, using='default')) == 2


def test_duplicate_celery_deliveries_publish_manifest_once(upload):
    from aquillm.tasks import ingest_uploaded_file_task
    from apps.ingestion.models import IngestionOutput
    item = upload()
    def run(_):
        close_old_connections()
        try:
            ingest_uploaded_file_task.run(item.pk)
        finally:
            close_old_connections()
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(run, range(2)))
    item.refresh_from_db()
    assert item.status == 'success'
    assert len(item.document_ids) == IngestionOutput.objects.count() == RawTextDocument.objects.count() == 2
