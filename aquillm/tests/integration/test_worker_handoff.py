"""Shared facade and beat contracts exercised through normal startup and Redis."""
import base64
import json
from datetime import timedelta
from unittest.mock import patch
from uuid import UUID

import pytest
from django.contrib.auth.models import User
from django.db import transaction
from django.utils import timezone

from .worker_handoff_support import WorkerHarness, isolated_broker, wait_until

pytestmark = pytest.mark.django_db(transaction=True)


def test_normal_discovery_registers_dense_task_and_gates_profile_recovery(tmp_path):
    harness = WorkerHarness(tmp_path)
    disabled = harness.run("discover", enabled=False)
    assert "documents.build_embedding_generation" in disabled["discovered"]
    assert disabled["schedule"] == {}
    enabled = harness.run("discover")
    assert enabled["schedule"]["profile-memory-reconciliation-recovery"] == {
        "task": "aquillm.tasks.recover_profile_memory_reconciliation_jobs",
        "schedule": 60, "kwargs": {"limit": 25}, "options": {"queue": "celery"},
    }


def test_committed_generation_is_consumed_without_changing_source_or_citations(tmp_path):
    harness = WorkerHarness(tmp_path)
    harness.run("discover")  # No task import before normal fresh-process registry proof.
    from apps.collections.models import Collection
    from apps.documents.models import RawTextDocument, TextChunk
    from apps.documents.models.embedding_index import (
        ChunkEmbedding, DocumentEmbeddingState, EmbeddingGeneration,
    )
    from apps.documents.services import embedding_index_store as store
    from apps.documents.services import representation_store as representations
    from apps.documents.services.representation_contracts import DocumentRefV1
    from apps.documents.tasks.embedding_index import rebuild_document_embeddings
    from lib.embeddings.document_provider import compatible_receipt
    from lib.embeddings.document_binding import validate_stored_binding
    from lib.embeddings.provenance import input_digest
    from .worker_handoff_provider import verified_space

    user = User.objects.create(username="handoff-dense")
    collection = Collection.objects.create(name="Synthetic handoff")
    doc = RawTextDocument(title="Synthetic", full_text="source", collection=collection,
                          ingested_by=user)
    doc.save(dont_rechunk=True)
    chunk = TextChunk.objects.bulk_create([TextChunk(doc_id=doc.id, content="source",
        start_position=0, end_position=6, chunk_number=0)])[0]
    ref = DocumentRefV1(doc._meta.label_lower, doc.pkid, doc.id, collection.pk, doc.full_text_hash)
    selection = representations.freeze_active_representations((ref,), using="default").selections[0]
    before = list(TextChunk.objects.values("pk", "doc_id", "content", "start_position",
                                         "end_position", "chunk_number"))
    original = (doc.id, doc.pkid, doc.full_text, doc.full_text_hash, doc.ingestion_complete)
    space = verified_space()
    store.persist_embedding_space(space, using="default")
    with isolated_broker(harness):
        with transaction.atomic():
            key = rebuild_document_embeddings(selection, space_key=space.space_key,
                                              using="default", timeout_ms=120000)
            assert isinstance(key, str) and str(UUID(key)) == key
            assert harness.queued(harness.queue) == []
        messages = harness.queued(harness.queue)
        assert len(messages) == 1
        assert messages[0]["headers"]["task"] == "documents.build_embedding_generation"
        args, kwargs, _ = json.loads(base64.b64decode(messages[0]["body"]))
        assert args == [key] and kwargs == {"using": "default", "timeout_ms": 120000}
        with harness.worker() as process:
            wait_until(lambda: EmbeddingGeneration.objects.filter(pk=key, state="ready").exists(),
                       process=process, log=harness.log)
        generation = EmbeddingGeneration.objects.get(pk=key)
        assert generation.deliveries == 1
        assert DocumentEmbeddingState.objects.get().active_generation_id == generation.pk
        sidecar = ChunkEmbedding.objects.get(generation=generation, chunk=chunk)
        assert compatible_receipt(sidecar.vector, sidecar.receipt, space=space,
            role="search_document", prepared_input_sha256=input_digest("source")) == space.space_key
        validate_stored_binding(sidecar.serving_binding, sidecar.vector, sidecar.receipt,
                               space=space, modality="text", media_sha256=None)
        assert sidecar.serving_binding["space_key"] == space.space_key
        assert sidecar.serving_binding["prepared_input_sha256"] == input_digest("source")
        assert list(TextChunk.objects.values(*before[0])) == before
        doc.refresh_from_db()
        assert (doc.id, doc.pkid, doc.full_text, doc.full_text_hash, doc.ingestion_complete) == original
        assert doc.ingestion_complete


def test_configured_due_tick_recovers_25_retained_jobs_and_preserves_unknown_coverage(tmp_path):
    harness = WorkerHarness(tmp_path)
    harness.run("discover")
    from apps.memory.models import UserMemoryFact, ProfileMemoryReconciliationJob
    from apps.memory.services.profile_facts import revoke_profile_fact
    from aquillm.tasks import reconcile_profile_memory_task
    user = User.objects.create(username="handoff-memory")
    facts = []
    with isolated_broker(harness):
        # Publication fails at the external broker boundary after the erasure commits.
        with patch.object(reconcile_profile_memory_task, "apply_async", side_effect=ConnectionError("synthetic broker unavailable")):
            fact = UserMemoryFact.objects.create(user=user, fact="Synthetic preference",
                origin="manual", provider_memory_ids=["known-0"], provider_receipt_complete=True,
                reconciliation_status="linked")
            revoke_profile_fact(user=user, fact_id=fact.pk, expected_revision=1)
        fact.refresh_from_db()
        assert fact.fact == "" and fact.status == "revoked"
        assert ProfileMemoryReconciliationJob.objects.get(fact=fact).state == "pending"
        assert harness.queued("memory-promotion") == []
        facts.append(fact)
        for index in range(1, 26):
            fact = UserMemoryFact.objects.create(user=user, fact="", status="revoked", origin="manual",
                provider_memory_ids=[f"known-{index}"], provider_receipt_complete=index != 1,
                reconciliation_status="pending")
            ProfileMemoryReconciliationJob.objects.create(fact=fact)
            facts.append(fact)
        for index, fact in enumerate(facts):
            ProfileMemoryReconciliationJob.objects.filter(fact=fact).update(
                next_attempt_at=timezone.now() - timedelta(seconds=100 - index))
        receipt = harness.run("schedule")
        assert receipt["published"] == 1
        messages = harness.queued("celery")
        assert len(messages) == 1
        assert messages[0]["headers"]["task"] == "aquillm.tasks.recover_profile_memory_reconciliation_jobs"
        args, kwargs, _ = json.loads(base64.b64decode(messages[0]["body"]))
        assert args == [] and kwargs == {"limit": 25}
        with harness.worker() as process:
            wait_until(lambda: ProfileMemoryReconciliationJob.objects.filter(state__in=["complete", "failed"]).count() == 25,
                       process=process, log=harness.log)
        assert set(harness.deleted()) == {f"known-{index}" for index in range(25)}
        assert len(harness.deleted()) == 25
        for fact in facts[:25]:
            fact.refresh_from_db()
            job = ProfileMemoryReconciliationJob.objects.get(fact=fact)
            assert job.attempts == 1 and fact.fact == ""
            if fact.pk == facts[1].pk:
                assert job.state == fact.reconciliation_status == "failed"
                assert job.last_error == "unidentified_provider_copies"
            else:
                assert job.state == fact.reconciliation_status == "complete"
        remaining = ProfileMemoryReconciliationJob.objects.get(fact=facts[25])
        assert remaining.state == "pending" and remaining.attempts == 0
