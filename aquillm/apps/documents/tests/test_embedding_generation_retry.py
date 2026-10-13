"""Durable delivery ordinals survive explicit recovery in fresh Celery messages."""

from datetime import timedelta
from unittest.mock import patch

import pytest
from celery.exceptions import Retry
from django.utils import timezone

from apps.documents.models.embedding_index import EmbeddingGeneration
from apps.documents.services import embedding_index_delivery as delivery
from apps.documents.tasks import embedding_index as task
from apps.documents.tests.test_embedding_generation import legacy as legacy
from apps.documents.tests.test_embedding_generation_deadlines import schedule
from lib.embeddings.document_contracts import build_budget
from lib.embeddings.errors import EmbeddingUpstreamUnavailableError
from lib.embeddings.tests.test_document_provider import transport as transport

pytestmark = pytest.mark.django_db(transaction=True)


def test_expiry_resume_outage_retry_obeys_durable_fence(legacy, transport, monkeypatch):
    selection, key = schedule(legacy, transport)
    clock, wall = [0.0], [timezone.now()]
    monkeypatch.setattr(delivery.timezone, "now", lambda: wall[0])
    monkeypatch.setattr(
        task,
        "build_budget",
        lambda timeout_ms: build_budget(timeout_ms, clock=lambda: clock[0]),
    )
    original = task.embed_document_batch
    calls = []

    def provider(*args, **kwargs):
        calls.append(True)
        if len(calls) == 1:
            clock[0] = 121.0
            raise TimeoutError("synthetic first delivery expiry")
        if len(calls) == 2:
            raise EmbeddingUpstreamUnavailableError("synthetic resumed outage")
        return original(*args, **kwargs)

    monkeypatch.setattr(task, "embed_document_batch", provider)
    with pytest.raises(TimeoutError):
        task.build_embedding_generation.run(key)
    row = EmbeddingGeneration.objects.get(pk=key)
    assert (row.deliveries, row.state) == (1, "building")
    wall[0] = row.next_attempt_at
    with patch.object(task.build_embedding_generation, "delay") as enqueue:
        assert (
            task.rebuild_document_embeddings(
                selection, space_key=transport[0].space_key, using="default"
            )
            == key
        )
        enqueue.assert_called_once_with(key, using="default", timeout_ms=120000)

    task.build_embedding_generation.push_request(retries=0)
    try:
        with patch.object(
            task.build_embedding_generation, "retry", side_effect=Retry()
        ) as retry:
            with pytest.raises(Retry):
                task.build_embedding_generation.run(key)
        delay = retry.call_args.kwargs["countdown"]
    finally:
        task.build_embedding_generation.pop_request()
    row.refresh_from_db()
    assert row.deliveries == 2
    scheduled_at = wall[0] + timedelta(seconds=delay)
    assert scheduled_at >= row.next_attempt_at
    wall[0] = scheduled_at
    task.build_embedding_generation.push_request(retries=1)
    try:
        assert task.build_embedding_generation.run(key) == "activated"
    finally:
        task.build_embedding_generation.pop_request()
    row.refresh_from_db()
    assert (row.deliveries, row.state, len(calls)) == (3, "ready", 3)
