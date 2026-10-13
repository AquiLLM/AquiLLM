"""Stored full-text vectors require current exact source projection bindings."""

import pytest
from django.db import transaction

from apps.documents.models import TextChunk
from apps.documents.tests.test_embedding_generation import legacy as legacy
from .representation_support import retained as retained, upload as upload

pytestmark = pytest.mark.django_db(transaction=True)


def publish(legacy):
    from apps.documents.services.lexical_publication import publish_chunk_search_vectors

    doc, _ = legacy
    with transaction.atomic():
        locked = type(doc).objects.select_for_update().get(pk=doc.pk)
        publish_chunk_search_vectors(
            document=locked,
            chunks=tuple(TextChunk.objects.filter(doc_id=doc.id)),
            using="default",
        )


def test_projection_publication_preserves_citation_and_binds_title(legacy):
    from apps.documents.services.chunk_search_scope import freeze_search_scope
    from apps.documents.services.lexical_readiness import full_text_candidates

    before = tuple(
        TextChunk.objects.values_list("pk", "content", "start_position", "end_position")
    )
    publish(legacy)
    scope = freeze_search_scope((legacy[0],), using="default")
    rows, coverage = full_text_candidates(TextChunk, "Synthetic", scope, limit=12)
    assert len(rows) == 1
    assert coverage.status == "ready"
    assert coverage.indexed_modalities == ("text",)
    assert (
        tuple(
            TextChunk.objects.values_list(
                "pk", "content", "start_position", "end_position"
            )
        )
        == before
    )
    type(legacy[0]).objects.filter(pk=legacy[0].pk).update(title="Revised")
    rows, coverage = full_text_candidates(TextChunk, "Synthetic", scope, limit=12)
    assert rows == ()
    assert coverage.status == "incomplete"


def test_nullable_or_stale_vector_is_not_coverage(legacy):
    from apps.documents.services.chunk_search_scope import freeze_search_scope
    from apps.documents.services.lexical_readiness import full_text_candidates

    scope = freeze_search_scope((legacy[0],), using="default")
    assert (
        full_text_candidates(TextChunk, "source", scope, limit=12)[1].status
        == "incomplete"
    )
    publish(legacy)
    TextChunk.objects.update(content="mutated")
    assert (
        full_text_candidates(TextChunk, "source", scope, limit=12)[1].status
        == "incomplete"
    )


def test_publication_requires_existing_transaction(legacy):
    from apps.documents.services.lexical_publication import publish_chunk_search_vectors

    with pytest.raises(ValueError):
        publish_chunk_search_vectors(
            document=legacy[0], chunks=tuple(TextChunk.objects.all()), using="default"
        )


def test_mixed_modality_is_explicitly_incomplete(legacy):
    from apps.documents.services.chunk_search_scope import freeze_search_scope
    from apps.documents.services.lexical_readiness import full_text_candidates

    publish(legacy)
    TextChunk.objects.bulk_create(
        [
            TextChunk(
                doc_id=legacy[0].id,
                content="caption",
                chunk_number=1,
                start_position=7,
                end_position=14,
                modality="image",
            )
        ]
    )
    scope = freeze_search_scope((legacy[0],), using="default")
    rows, coverage = full_text_candidates(TextChunk, "source", scope, limit=12)
    assert rows == ()
    assert coverage.status == "unsupported_modality"
    assert coverage.unsupported_count == 1


@pytest.mark.parametrize("mutation", ["section", "kind", "vector", "extra_binding"])
def test_projection_tampering_cannot_reuse_nonnull_fts(legacy, mutation):
    from django.contrib.postgres.search import SearchVector
    from django.db.models import Value

    from apps.documents.services.chunk_search_scope import freeze_search_scope
    from apps.documents.services.lexical_readiness import full_text_candidates

    publish(legacy)
    if mutation == "section":
        TextChunk.objects.update(section_path=["Unverified heading"])
    elif mutation == "kind":
        TextChunk.objects.update(source_kind="bibliography")
    elif mutation == "vector":
        TextChunk.objects.update(search_vector=SearchVector(Value("substitute")))
    else:
        binding = TextChunk.objects.get().search_vector_binding
        TextChunk.objects.update(search_vector_binding={**binding, "extra": "unknown"})
    scope = freeze_search_scope((legacy[0],), using="default")
    assert (
        full_text_candidates(TextChunk, "source", scope, limit=12)[1].status
        == "incomplete"
    )


def test_failed_projection_rolls_back_chunk_publication(retained, monkeypatch):
    from apps.documents.models import ChunkPublication, DocumentRepresentationState
    from apps.documents.services import lexical_publication
    from apps.documents.tasks.chunking import create_chunks
    from .test_representation_tasks import delivery

    document, _, old = retained
    envelope = delivery(document)
    original_source = (document.full_text, document.full_text_hash)
    original_binding = old.search_vector_binding

    def fail(*args, **kwargs):
        raise ValueError("synthetic projection failure")

    monkeypatch.setattr(lexical_publication, "publish_chunk_search_vectors", fail)
    with pytest.raises(ValueError, match="projection failure"):
        create_chunks.run(**envelope)
    document.refresh_from_db()
    state = DocumentRepresentationState.objects.get(document_id=document.id)
    assert state.active_generation_id is None
    assert (document.full_text, document.full_text_hash) == original_source
    assert list(document.chunks.values_list("pk", flat=True)) == [old.pk]
    saved = TextChunk.objects.get(pk=old.pk)
    assert saved.content == old.content and saved.search_vector_binding == original_binding
    failed = state.generations.get()
    assert failed.status == "failed" and failed.textchunk_set.exists()
    assert not failed.textchunk_set.filter(search_vector_binding__isnull=False).exists()
    intent = ChunkPublication.objects.get(pk=envelope["publication_id"])
    assert intent.failure_kind and intent.attempts == envelope["publication_attempt"]
    assert not document.ingestion_complete


def test_future_staged_projection_uses_validated_structure_without_pointer_activation(
    retained,
):
    from apps.documents.services.lexical_publication import publish_chunk_search_vectors
    from .lexical_owner_support import staged_projection
    document, _, old = retained
    original_text = document.full_text
    _, generation, extraction = staged_projection(retained)
    chunk = generation.textchunk_set.get(start_position=9)
    with transaction.atomic():
        publish_chunk_search_vectors(
            document=document,
            chunks=(chunk,),
            using="default",
            canonical_text=extraction.canonical_text,
            representation_key=generation.pk,
            source_revision_sha256=generation.source_revision_sha256,
            extraction=extraction,
        )
    chunk.refresh_from_db()
    assert chunk.source_kind == "body"
    assert chunk.section_path == ["Measured"]
    assert chunk.search_vector_binding["representation_key"] == str(generation.pk)
    assert chunk.search_vector_binding["source_revision_sha256"] == generation.source_revision_sha256
    assert chunk.content == extraction.canonical_text[9:]
    document.refresh_from_db()
    assert document.full_text == original_text
    assert generation.owner.active_generation_id is None
    assert list(document.chunks.values_list("pk", flat=True)) == [old.pk]


def test_staged_binding_uses_post_update_owner_hash_and_separate_input_revision(retained):
    from apps.documents.services.lexical_projection import text_hash
    from apps.documents.services.representation_activation import activate_representation
    from .lexical_owner_support import staged_projection
    doc, _, old = retained
    original_hash = doc.full_text_hash
    lease, generation, extraction = staged_projection(retained)
    assert activate_representation(lease, expected_active_key=None, using="default") == "activated"
    chunk = generation.textchunk_set.get(start_position=9)
    assert generation.predecessor_source_hash == original_hash
    assert chunk.search_vector_binding["document"]["source_hash"] == text_hash(extraction.canonical_text)
    assert chunk.search_vector_binding["source_revision_sha256"] == generation.extraction_revision.manifest_sha256
    assert chunk.search_vector_binding["source_revision_sha256"] != original_hash
    assert TextChunk.objects.get(pk=old.pk).content == old.content
