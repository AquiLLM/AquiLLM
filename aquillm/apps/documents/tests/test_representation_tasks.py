"""Real chunk-outbox deliveries publish strict representations without legacy vectors."""
from unittest.mock import patch

import pytest

from apps.documents.models import ChunkPublication, DocumentRepresentationState, TextChunk
from apps.documents.tasks.chunking import create_chunks
from apps.documents.services.representation_store import freeze_active_representations, active_chunk_scope
from .representation_support import retained as retained, upload as upload, reference

pytestmark = pytest.mark.django_db(transaction=True)


def delivery(document):
    row = ChunkPublication.objects.get(document_id=document.id)
    return dict(doc_id=str(document.id), expected_source_hash=document.full_text_hash,
        concrete_model_label=document._meta.label_lower, document_pkid=document.pkid,
        publication_id=row.pk, publication_generation=str(row.generation), publication_attempt=row.attempts)


def test_actual_outbox_delivery_activates_structural_generation_and_exact_replay(retained):
    doc, output, old = retained
    envelope = delivery(doc)
    with patch("apps.documents.tasks.chunking._embed_chunks", side_effect=AssertionError("legacy embedding writer")):
        assert create_chunks.run(**envelope) == "activated"
        assert create_chunks.run(**envelope) == "stale"
    doc.refresh_from_db()
    state = DocumentRepresentationState.objects.get(document_id=doc.id)
    generation = state.active_generation
    assert generation.origin_output_id == output.pk
    assert generation.publication_attempt == envelope["publication_attempt"]
    assert generation.extraction_revision_id == output.extraction_revision_id
    assert not ChunkPublication.objects.filter(pk=envelope["publication_id"]).exists()
    frozen = freeze_active_representations((reference(doc),), using="default")
    rows = list(active_chunk_scope(TextChunk.objects.filter(doc_id=doc.id), snapshot=frozen))
    assert rows and all(row.representation_generation_id == generation.pk for row in rows)
    assert all(row.embedding is None and row.search_vector is not None for row in rows)
    assert old.pk not in {row.pk for row in rows}
    assert TextChunk.objects.get(pk=old.pk).content == old.content


def test_wrong_outbox_attempt_never_writes_generation(retained):
    doc, _, _ = retained
    envelope = delivery(doc)
    envelope["publication_attempt"] += 1
    assert create_chunks.run(**envelope) == "stale"
    assert not DocumentRepresentationState.objects.filter(document_id=doc.id).exists()


def test_activation_failure_rolls_back_pointer_and_preserves_exact_failed_intent(retained):
    from apps.documents.services import representation_activation
    from apps.knowledge_graph.models import GraphRebuildRequest
    doc, _, old = retained
    envelope = delivery(doc)
    before = list(GraphRebuildRequest.objects.values_list("pk", flat=True))
    with patch.object(representation_activation, "finish_document_representation_transition", side_effect=RuntimeError("commit barrier")):
        with pytest.raises(RuntimeError, match="commit barrier"):
            create_chunks.run(**envelope)
    state = DocumentRepresentationState.objects.get(document_id=doc.id)
    assert state.active_generation_id is None
    assert list(GraphRebuildRequest.objects.values_list("pk", flat=True)) == before
    assert TextChunk.objects.get(pk=old.pk).content == old.content
    intent = ChunkPublication.objects.get(pk=envelope["publication_id"])
    assert intent.failure_kind and intent.attempts == envelope["publication_attempt"]


def test_death_after_commit_keeps_citations_and_replay_validates_registered_input(retained):
    doc, _, old = retained
    envelope = delivery(doc)
    with patch("apps.documents.tasks.chunking.notify_ingest_monitor_complete", side_effect=SystemExit("worker died")):
        with pytest.raises(SystemExit):
            create_chunks.run(**envelope)
    assert not ChunkPublication.objects.filter(pk=envelope["publication_id"]).exists()
    doc.refresh_from_db()
    before = list(doc.chunks.values_list("pk", flat=True))
    assert before and old.pk not in before
    assert create_chunks.run(**envelope) == "stale"
    assert create_chunks.run(str(doc.id), doc.full_text_hash, doc._meta.label_lower, doc.pkid) == "already_committed"
    assert list(doc.chunks.values_list("pk", flat=True)) == before


def test_unfenced_success_ack_cannot_delete_a_newer_attempt(retained):
    from apps.documents.services import representation_replay
    from django.utils import timezone
    from datetime import timedelta
    from uuid import uuid4
    doc, _, _ = retained
    assert create_chunks.run(**delivery(doc)) == "activated"
    doc.refresh_from_db()
    pending = ChunkPublication.objects.create(concrete_model_label=doc._meta.label_lower,
        document_pkid=doc.pkid, document_id=doc.id, source_hash=doc.full_text_hash,
        collection_ids=[doc.collection_id], generation=uuid4(), attempts=1,
        next_attempt_at=timezone.now() + timedelta(seconds=900))
    original = representation_replay.replay_registered_representation
    def advance(*args, **kwargs):
        ChunkPublication.objects.filter(pk=pending.pk).update(attempts=2)
        return original(*args, **kwargs)
    with patch.object(representation_replay, "replay_registered_representation", side_effect=advance):
        assert create_chunks.run(str(doc.id), doc.full_text_hash, doc._meta.label_lower, doc.pkid) == "already_committed"
    pending.refresh_from_db()
    assert pending.attempts == 2 and not pending.failure_kind


def test_unfenced_delivery_uses_canonical_snapshot_without_manufacturing_artifact(retained):
    from apps.documents.models import RawTextDocument
    doc, _, _ = retained
    with patch.object(create_chunks, "delay"):
        manual = RawTextDocument.objects.create(title="Manual", full_text="canonical user source",
            collection=doc.collection, ingested_by=doc.ingested_by)
    assert create_chunks.run(str(manual.id), manual.full_text_hash, manual._meta.label_lower, manual.pkid) == "activated"
    state = DocumentRepresentationState.objects.get(document_id=manual.id)
    assert state.active_generation.input_kind == "canonical_snapshot"
    assert state.active_generation.source_snapshot.source_artifact_id is None
    assert not ChunkPublication.objects.filter(document_id=manual.id).exists()


def test_duplicate_delivery_resumes_staged_generation_after_precommit_death(retained):
    from apps.documents.tasks import representation
    doc, _, _ = retained
    envelope = delivery(doc)
    with patch.object(representation, "activate_representation", side_effect=SystemExit("worker died")):
        with pytest.raises(SystemExit):
            create_chunks.run(**envelope)
    state = DocumentRepresentationState.objects.get(document_id=doc.id)
    staged = state.generations.get()
    assert staged.status == "staged" and state.active_generation_id is None
    before = list(staged.textchunk_set.values_list("pk", flat=True))
    claim = (staged.lease_token, staged.lease_expires_at, staged.attempt)
    assert create_chunks.run(**envelope) == "activated"
    state.refresh_from_db()
    assert state.active_generation_id == staged.pk and state.generations.count() == 1
    assert list(staged.textchunk_set.values_list("pk", flat=True)) == before
    staged.refresh_from_db()
    assert (staged.lease_token, staged.lease_expires_at, staged.attempt) == claim


def test_expired_staged_attempt_recovers_through_new_outbox_attempt(retained):
    from datetime import timedelta
    from apps.documents.tasks import representation
    from apps.documents.services.chunk_publication import recover_due_chunk_publications
    from apps.documents.services.representation_activation import activate_representation
    from apps.documents.services.representation_contracts import BuildLeaseV1
    doc, _, old = retained
    envelope = delivery(doc)
    with patch.object(representation, "activate_representation", side_effect=SystemExit("worker died")):
        with pytest.raises(SystemExit):
            create_chunks.run(**envelope)
    state = DocumentRepresentationState.objects.get(document_id=doc.id)
    staged = state.generations.get()
    claim = (staged.lease_token, staged.lease_expires_at, staged.attempt)
    now = staged.lease_expires_at + timedelta(seconds=1)
    with patch("django.utils.timezone.now", return_value=now), patch.object(create_chunks, "delay") as broker:
        assert activate_representation(BuildLeaseV1(staged.pk, staged.lease_token),
            expected_active_key=None, using="default") == "stale"
        assert recover_due_chunk_publications() == 1
        broker.assert_called_once()
        assert broker.call_args.kwargs["publication_attempt"] == envelope["publication_attempt"] + 1
        assert create_chunks.run(**envelope) == "stale"
        assert create_chunks.run(**delivery(doc)) == "activated"
    state.refresh_from_db()
    staged.refresh_from_db()
    assert state.active_generation_id != staged.pk and state.generations.count() == 2
    assert state.active_generation.publication_attempt == envelope["publication_attempt"] + 1
    assert (staged.lease_token, staged.lease_expires_at, staged.attempt) == claim
    assert staged.status == "staged" and TextChunk.objects.filter(pk=old.pk).exists()
