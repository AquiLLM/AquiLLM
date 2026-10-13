"""Committed output identity survives failure, replay and foreign collisions."""
import pytest
from apps.documents.models import RawTextDocument
from aquillm.task_ingest_uploaded import run_ingest_uploaded_file
from .output_support import upload

pytestmark = pytest.mark.django_db(transaction=True)


def test_second_failure_preserves_tracked_first_output(upload, monkeypatch):
    item = upload()
    original = RawTextDocument.save
    def save(document, *args, **kwargs):
        if document.full_text == 'second body':
            raise ValueError('private parser details must not leak')
        return original(document, *args, **kwargs)
    monkeypatch.setattr(RawTextDocument, 'save', save)
    run_ingest_uploaded_file(item.pk)
    item.refresh_from_db()
    assert item.status == 'partial'
    assert item.document_ids == [str(RawTextDocument.objects.get().id)]
    assert [o['status'] for o in item.parser_metadata['outputs']] == ['success', 'error']
    assert 'private parser' not in str(item.parser_metadata)
    before = item.document_ids[:]
    run_ingest_uploaded_file(item.pk)
    item.refresh_from_db()
    assert item.document_ids == before
    assert RawTextDocument.objects.count() == 1


def test_repeated_success_is_noop(upload):
    item = upload()
    run_ingest_uploaded_file(item.pk)
    item.refresh_from_db()
    before = item.document_ids[:]
    run_ingest_uploaded_file(item.pk)
    item.refresh_from_db()
    assert item.document_ids == before
    assert RawTextDocument.objects.count() == 2
    assert item.status == 'success'


def test_identical_foreign_source_is_explicit_collision(upload):
    item = upload((('first.txt', b'same'), ('second.txt', b'same')))
    run_ingest_uploaded_file(item.pk)
    item.refresh_from_db()
    assert item.status == 'partial'
    assert RawTextDocument.objects.count() == 1
    assert item.parser_metadata['outputs'][1]['error_code'] == 'content_hash_collision'
    assert RawTextDocument.objects.get().full_text == 'same'


def test_checkpoint_keeps_chunk_outbox_and_separate_index_status(upload):
    from apps.documents.models.chunk_publication import ChunkPublication
    item = upload((('first.txt', b'first'),))
    run_ingest_uploaded_file(item.pk)
    item.refresh_from_db()
    document = RawTextDocument.objects.get()
    intent = ChunkPublication.objects.get(document_id=document.id)
    assert intent.document_pkid == document.pkid
    assert intent.source_hash == document.full_text_hash
    assert item.parser_metadata['outputs'][0]['status'] == 'success'
    assert item.parser_metadata['outputs'][0]['index_status'] == 'pending'
