"""Periodic redelivery of building generations whose delivery lapsed without a fence."""
from datetime import timedelta
from unittest.mock import patch

import httpx
import pytest
from django.utils import timezone

from apps.documents.models.embedding_index import EmbeddingGeneration
from apps.documents.services import embedding_index_store as store
from apps.documents.services.embedding_index_delivery import recover_due_embedding_generations
from apps.documents.tasks import embedding_index as task
from lib.embeddings.utils import EmbeddingContractError
from .test_chunk_publication_recovery import document_factory as document_factory
from .embedding_outage_support import (
    dense_source as dense_source, legacy as legacy, transport as transport,
    schedule, transport_failure, assert_dense_ready,
)

pytestmark = pytest.mark.django_db(transaction=True)


def recover(limit=3):
    with patch.object(task.build_embedding_generation, "delay") as queue:
        count = recover_due_embedding_generations(limit=limit, using="default")
    assert count == queue.call_count
    return [c.args[0] for c in queue.call_args_list]


def lapse(key, minutes=6):
    assert EmbeddingGeneration.objects.filter(pk=key).update(
        next_attempt_at=timezone.now() - timedelta(minutes=minutes)) == 1


def test_lapsed_delivery_is_redelivered_once_and_completes(dense_source):
    key = schedule(dense_source)
    # A worker claimed the lease, then hit its deadline without recording a failure.
    assert store.claim_embedding_delivery(key, using="default") is not None
    assert recover() == []
    lapse(key, minutes=1)
    assert recover() == []  # lapsed, but still inside the quiet period
    lapse(key)
    assert recover() == [str(key)]
    assert recover() == []  # the queued redelivery is not sent again
    assert task.build_embedding_generation.run(str(key)) == "activated"
    assert_dense_ready(key, dense_source)
    assert EmbeddingGeneration.objects.get(pk=key).deliveries == 2
    lapse(key)
    assert recover() == []


def test_lost_first_dispatch_is_redelivered_only_after_quiet_period(dense_source):
    key = schedule(dense_source)
    assert recover() == []
    EmbeddingGeneration.objects.filter(pk=key).update(created_at=timezone.now() - timedelta(minutes=6))
    assert recover() == [str(key)]
    assert recover() == []


def test_terminal_failure_is_never_redelivered(dense_source, monkeypatch):
    key = schedule(dense_source)
    transport_failure(monkeypatch, httpx.UnsupportedProtocol("synthetic configuration"))
    with pytest.raises(EmbeddingContractError):
        task.build_embedding_generation.run(str(key))
    lapse(key)
    assert recover() == []


def slow_batches(monkeypatch, seconds):
    """Each embedding batch uses `seconds` of a fake build clock."""
    from lib.embeddings.document_contracts import build_budget
    now = [1000.0]
    monkeypatch.setattr(task, "build_budget", lambda timeout_ms=120000: build_budget(timeout_ms, clock=lambda: now[0]))
    original = task.embed_document_batch

    def embed(*args, **kwargs):
        results = original(*args, **kwargs)
        now[0] += seconds
        return results
    monkeypatch.setattr(task, "embed_document_batch", embed)


def test_progress_near_deadline_hands_off_without_spending_a_delivery(dense_source, monkeypatch):
    key = schedule(dense_source)
    slow_batches(monkeypatch, 110)
    with patch.object(task.build_embedding_generation, "delay") as queue:
        assert task.build_embedding_generation.run(str(key)) == "handed_off"
    queue.assert_called_once_with(str(key), using="default", timeout_ms=120000)
    row = EmbeddingGeneration.objects.get(pk=key)
    assert (row.state, row.failure_kind, row.deliveries) == ("building", "", 0)
    assert row.embeddings.count() == row.expected_count
    assert task.build_embedding_generation.run(str(key)) == "activated"
    assert_dense_ready(key, dense_source)


def test_deadline_without_progress_still_spends_the_delivery(dense_source, monkeypatch):
    key = schedule(dense_source)
    slow_batches(monkeypatch, 130)
    with patch.object(task.build_embedding_generation, "delay") as queue:
        with pytest.raises(TimeoutError):
            task.build_embedding_generation.run(str(key))
    queue.assert_not_called()
    row = EmbeddingGeneration.objects.get(pk=key)
    assert (row.state, row.deliveries) == ("building", 1)
    assert not row.embeddings.exists()


def test_recovery_schedule_entry():
    from aquillm.celery_schedules import application_maintenance_schedule
    entry = application_maintenance_schedule(enabled=True)["document-embedding-recovery"]
    assert entry["task"] == "apps.documents.tasks.embedding_recovery.recover_embedding_generations"
    assert (entry["schedule"], entry["kwargs"], entry["options"]["queue"]) == (60, {"limit": 3}, "celery")
