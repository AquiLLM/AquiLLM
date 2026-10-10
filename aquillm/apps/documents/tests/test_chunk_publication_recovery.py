"""Durable chunk publication survives broker failure, rollback, and stale delivery."""
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor
from importlib import import_module
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from django.apps import apps
from django.contrib.auth.models import User
from django.db import close_old_connections, transaction
from django.utils import timezone

from apps.collections.models import Collection
from apps.documents.models import RawTextDocument
from apps.documents.tasks.chunking import create_chunks

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.fixture
def document_factory():
    user = User.objects.create(username='chunk-recovery')
    collection = Collection.objects.create(name='Recovery')

    def create(text='Source text'):
        return RawTextDocument.objects.create(title='Source', full_text=text, collection=collection, ingested_by=user)
    return create


def intent_model():
    return apps.get_model('apps_documents', 'ChunkPublication')


def test_broker_failure_keeps_durable_intent_and_recovers_once(document_factory):
    with patch.object(create_chunks, 'delay', side_effect=ConnectionError('broker offline')):
        document = document_factory()
    intent = intent_model().objects.get(document_id=document.id)
    assert intent.last_error == 'ConnectionError'
    assert intent.next_attempt_at > timezone.now()
    from apps.documents.tasks.chunk_recovery import recover_chunk_publications
    intent_model().objects.filter(pk=intent.pk).update(next_attempt_at=timezone.now() - timedelta(seconds=1))
    with patch.object(create_chunks, 'delay') as queue:
        recover_chunk_publications.run()
        recover_chunk_publications.run()
    intent.refresh_from_db()
    queue.assert_called_once_with(str(document.id), document.full_text_hash, document._meta.label_lower, document.pkid,
                                  publication_id=intent.pk, publication_generation=str(intent.generation), publication_attempt=intent.attempts)
    document.refresh_from_db()
    assert not document.ingestion_complete


def test_outer_rollback_creates_neither_intent_nor_task(document_factory):
    with patch.object(create_chunks, 'delay') as queue:
        with pytest.raises(RuntimeError):
            with transaction.atomic():
                document_factory()
                raise RuntimeError('rollback')
    assert not intent_model().objects.exists()
    assert not RawTextDocument.objects.exists()
    queue.assert_not_called()


def test_deleted_document_never_publishes_saved_intent(document_factory):
    with patch.object(create_chunks, 'delay', side_effect=ConnectionError('offline')):
        document = document_factory()
    intent = intent_model().objects.get(document_id=document.id)
    document.delete()
    from apps.documents.services.chunk_publication import dispatch_chunk_publication
    with patch.object(create_chunks, 'delay') as queue:
        dispatch_chunk_publication(intent.pk)
    queue.assert_not_called()
    assert not intent_model().objects.filter(pk=intent.pk).exists()


def test_content_replacement_cannot_republish_old_snapshot(document_factory):
    with patch.object(create_chunks, 'delay', side_effect=ConnectionError('offline')):
        document = document_factory()
        old_intent = intent_model().objects.get(document_id=document.id)
        document.full_text = 'Replacement content'
        document.save(update_fields=['full_text'])
    from apps.documents.services.chunk_publication import dispatch_chunk_publication
    with patch.object(create_chunks, 'delay') as queue:
        dispatch_chunk_publication(old_intent.pk)
    queue.assert_not_called()
    assert not intent_model().objects.filter(pk=old_intent.pk).exists()
    assert intent_model().objects.filter(document_id=document.id, source_hash=document.full_text_hash).exists()


def test_due_scan_is_bounded_and_does_not_repeat_recently_queued_work(document_factory):
    with patch.object(create_chunks, 'delay', side_effect=ConnectionError('offline')):
        for index in range(3):
            document_factory(f'Document {index}')
    intent_model().objects.update(next_attempt_at=timezone.now() - timedelta(seconds=1))
    from apps.documents.tasks.chunk_recovery import recover_chunk_publications
    with patch.object(create_chunks, 'delay') as queue:
        recover_chunk_publications.run(limit=2)
        assert queue.call_count == 2
        recover_chunk_publications.run(limit=2)
        assert queue.call_count == 3


def test_successful_chunking_acknowledges_intent_and_redelivery_is_idempotent(document_factory, monkeypatch):
    from ._chunk_graph_lifecycle_support import configure_chunking_runtime, run_chunk_task
    chunking = configure_chunking_runtime(monkeypatch)
    with patch.object(create_chunks, 'delay'):
        document = document_factory()
    with patch('apps.knowledge_graph.services.builds.enqueue_document_build'):
        run_chunk_task(chunking, document)
        before = list(document.chunks.values_list('pk', flat=True))
        run_chunk_task(chunking, document)
    assert before and list(document.chunks.values_list('pk', flat=True)) == before
    assert not intent_model().objects.filter(document_id=document.id).exists()


def test_concurrent_dispatchers_publish_only_one_lease(document_factory):
    with patch.object(create_chunks, 'delay', side_effect=ConnectionError('offline')):
        document = document_factory()
    intent = intent_model().objects.get(document_id=document.id)
    intent_model().objects.filter(pk=intent.pk).update(next_attempt_at=timezone.now() - timedelta(seconds=1))
    from apps.documents.services.chunk_publication import dispatch_chunk_publication
    publishing, release = Event(), Event()

    def publish(*args, **kwargs):
        publishing.set()
        assert release.wait(5)

    def dispatch():
        close_old_connections()
        try:
            return dispatch_chunk_publication(intent.pk)
        finally:
            close_old_connections()

    with patch.object(create_chunks, 'delay', side_effect=publish) as queue:
        with ThreadPoolExecutor(max_workers=2) as workers:
            first = workers.submit(dispatch)
            try:
                assert publishing.wait(5)
                assert workers.submit(dispatch).result(timeout=5) is False
            finally:
                release.set()
            assert first.result(timeout=5) is True
    queue.assert_called_once()


def test_migration_recovers_valid_legacy_documents_once(document_factory):
    with patch.object(create_chunks, 'delay'):
        stranded = document_factory('Legacy extracted content')
        complete = document_factory('Already complete')
        blank = document_factory('   ')
    intent_model().objects.all().delete()
    RawTextDocument.objects.filter(pk=complete.pk).update(ingestion_complete=True)
    migration = import_module('apps.documents.migrations.0005_chunkpublication')
    editor = SimpleNamespace(connection=SimpleNamespace(alias='default'))
    with patch.object(create_chunks, 'delay') as queue:
        migration.backfill_incomplete_documents(apps, editor)
        migration.backfill_incomplete_documents(apps, editor)
    assert list(intent_model().objects.values_list('document_id', flat=True)) == [stranded.id]
    queue.assert_not_called()
