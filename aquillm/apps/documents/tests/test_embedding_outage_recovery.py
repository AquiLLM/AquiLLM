"""Exact-source durable outage, terminal failure, and reset regression tests."""
from datetime import timedelta
from unittest.mock import Mock, patch

import pytest
from django.utils import timezone

from aquillm.utils import EmbeddingUpstreamUnavailableError
from lib.embeddings.utils import EmbeddingContractError
from apps.documents.models import RawTextDocument, TextChunk
from apps.documents.models.chunk_publication import ChunkPublication
from apps.documents.services import chunk_publication as publication
from apps.documents.tasks.chunking import create_chunks
from .test_chunk_publication_recovery import document_factory
from ._chunk_graph_lifecycle_support import configure_chunking_runtime

pytestmark = pytest.mark.django_db(transaction=True)


def queued(document_factory):
    with patch.object(create_chunks, "delay") as queue:
        document = document_factory("Synthetic source " * 200)
    return document, queue.call_args


@pytest.mark.parametrize("failure", [EmbeddingUpstreamUnavailableError("offline"), EmbeddingContractError("malformed"), TypeError("bug")])
def test_document_batch_error_does_not_fan_out(document_factory, monkeypatch, failure):
    chunking = configure_chunking_runtime(monkeypatch)
    document, _ = queued(document_factory)
    batch = Mock(side_effect=failure)
    single = Mock()
    monkeypatch.setattr(chunking, "get_embeddings", batch)
    monkeypatch.setattr(TextChunk, "get_chunk_embedding", single)
    with pytest.raises(type(failure)):
        chunks, image = chunking._prepared_chunks(document)
        chunking._embed_chunks(document, chunks, image)
    assert batch.call_count == 1
    single.assert_not_called()


def test_contract_terminal_intent_reset_recovery_and_stale_failure(document_factory, monkeypatch):
    chunking = configure_chunking_runtime(monkeypatch)
    document, old_call = queued(document_factory)
    provider = chunking.get_embeddings
    monkeypatch.setattr(chunking, "get_embeddings", Mock(side_effect=EmbeddingContractError("malformed")))
    with pytest.raises(EmbeddingContractError):
        create_chunks.run(*old_call.args, **old_call.kwargs)
    intent = ChunkPublication.objects.get(document_id=document.id)
    assert intent.failure_kind == "contract"
    old_generation = intent.generation
    ChunkPublication.objects.filter(pk=intent.pk).update(next_attempt_at=timezone.now()-timedelta(seconds=1))
    with patch.object(create_chunks, "delay") as queue:
        assert publication.recover_due_chunk_publications() == 0
        queue.assert_not_called()
        assert publication.reset_chunk_publication(intent.pk, source_hash=document.full_text_hash, generation=old_generation)
    new_call = queue.call_args
    intent.refresh_from_db()
    assert intent.generation != old_generation
    assert not intent.failure_kind
    # Old delivery must not run the provider or poison the reset generation.
    chunking.get_embeddings.reset_mock()
    assert create_chunks.run(*old_call.args, **old_call.kwargs) == "stale"
    chunking.get_embeddings.assert_not_called()
    intent.refresh_from_db()
    assert not intent.failure_kind
    assert not publication.reset_chunk_publication(intent.pk, source_hash=document.full_text_hash, generation=old_generation)
    monkeypatch.setattr(chunking, "get_embeddings", provider)
    with patch("apps.knowledge_graph.services.builds.enqueue_document_build"):
        assert create_chunks.run(*new_call.args, **new_call.kwargs) == "committed"
    assert not ChunkPublication.objects.filter(pk=intent.pk).exists()
    document.refresh_from_db()
    assert document.ingestion_complete


def test_transient_exhaustion_retains_intent_and_recovers(document_factory, monkeypatch):
    chunking = configure_chunking_runtime(monkeypatch)
    document, call = queued(document_factory)
    provider = chunking.get_embeddings
    failing = Mock(side_effect=EmbeddingUpstreamUnavailableError("offline"))
    monkeypatch.setattr(chunking, "get_embeddings", failing)
    create_chunks.push_request(retries=2, called_directly=False)
    try:
        with pytest.raises(EmbeddingUpstreamUnavailableError):
            create_chunks.run(*call.args, **call.kwargs)
    finally:
        create_chunks.pop_request()
    assert failing.call_count == 1
    intent = ChunkPublication.objects.get(document_id=document.id)
    assert not intent.failure_kind
    assert intent.last_error == "EmbeddingUpstreamUnavailableError"
    ChunkPublication.objects.filter(pk=intent.pk).update(next_attempt_at=timezone.now()-timedelta(seconds=1))
    with patch.object(create_chunks, "delay") as queue:
        assert publication.recover_due_chunk_publications() == 1
    monkeypatch.setattr(chunking, "get_embeddings", provider)
    with patch("apps.knowledge_graph.services.builds.enqueue_document_build"):
        assert create_chunks.run(*queue.call_args.args, **queue.call_args.kwargs) == "committed"
    assert not ChunkPublication.objects.filter(pk=intent.pk).exists()


def test_legacy_failure_cannot_terminal_block_current_intent(document_factory, monkeypatch):
    chunking = configure_chunking_runtime(monkeypatch)
    document, _ = queued(document_factory)
    monkeypatch.setattr(chunking, "get_embeddings", Mock(side_effect=EmbeddingContractError("malformed")))
    with pytest.raises(EmbeddingContractError):
        create_chunks.run(str(document.id), document.full_text_hash, document._meta.label_lower, document.pkid)
    intent = ChunkPublication.objects.get(document_id=document.id)
    assert not intent.failure_kind


def test_same_source_save_does_not_clear_terminal_state(document_factory, monkeypatch):
    chunking = configure_chunking_runtime(monkeypatch)
    document, call = queued(document_factory)
    monkeypatch.setattr(chunking, "get_embeddings", Mock(side_effect=EmbeddingContractError("malformed")))
    with pytest.raises(EmbeddingContractError):
        create_chunks.run(*call.args, **call.kwargs)
    with patch.object(create_chunks, "delay") as queue:
        document.save()
    queue.assert_not_called()
    assert ChunkPublication.objects.get(document_id=document.id).failure_kind == "contract"
    with patch.object(create_chunks, "delay") as queue:
        document.full_text = "Changed synthetic source"
        document.save()
    assert queue.call_count == 1
    assert not ChunkPublication.objects.get(document_id=document.id).failure_kind


def test_document_worker_transient_retry_chain_is_three_attempts(document_factory, monkeypatch):
    from celery.exceptions import Retry
    chunking = configure_chunking_runtime(monkeypatch)
    document, call = queued(document_factory)
    provider = Mock(side_effect=EmbeddingUpstreamUnavailableError("offline"))
    monkeypatch.setattr(chunking, "get_embeddings", provider)
    for number in range(3):
        create_chunks.push_request(retries=number, called_directly=False, is_eager=True)
        try:
            with pytest.raises(Retry if number < 2 else EmbeddingUpstreamUnavailableError) as caught:
                create_chunks.run(*call.args, **call.kwargs)
            if number < 2:
                assert caught.value.when == 30 * 2 ** number
        finally:
            create_chunks.pop_request()
    assert provider.call_count == 3
    assert ChunkPublication.objects.filter(document_id=document.id).exists()


@pytest.mark.parametrize("change", ["reset", "lease", "source"])
def test_inflight_old_failure_cannot_block_new_generation(document_factory, monkeypatch, change):
    from uuid import uuid4
    chunking = configure_chunking_runtime(monkeypatch)
    document, call = queued(document_factory)
    def fail_after_change(*args, **kwargs):
        if change == "source":
            document.full_text = "New source while embedding"
            with patch.object(create_chunks, "delay"):
                document.save()
        else:
            # Concurrent reset / later dispatcher advancing the exact fence.
            intent = ChunkPublication.objects.get(document_id=document.id)
            updates = {"generation": uuid4()} if change == "reset" else {"attempts": intent.attempts + 1}
            ChunkPublication.objects.filter(pk=intent.pk).update(**updates)
        raise EmbeddingContractError("old failure")
    monkeypatch.setattr(chunking, "get_embeddings", fail_after_change)
    with pytest.raises(EmbeddingContractError):
        create_chunks.run(*call.args, **call.kwargs)
    assert not ChunkPublication.objects.get(document_id=document.id).failure_kind


def test_transient_peer_does_not_clear_terminal_failure(document_factory, monkeypatch):
    chunking = configure_chunking_runtime(monkeypatch)
    document, call = queued(document_factory)
    def transient_after_peer(*args, **kwargs):
        ChunkPublication.objects.filter(document_id=document.id).update(failure_kind="contract", last_error="EmbeddingContractError")
        raise EmbeddingUpstreamUnavailableError("offline")
    monkeypatch.setattr(chunking, "get_embeddings", transient_after_peer)
    with pytest.raises(EmbeddingUpstreamUnavailableError):
        create_chunks.run(*call.args, **call.kwargs)
    assert ChunkPublication.objects.get(document_id=document.id).failure_kind == "contract"
