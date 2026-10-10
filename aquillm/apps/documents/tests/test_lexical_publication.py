"""Stored full-text vectors require current exact source projection bindings."""

import pytest
from django.db import transaction

from apps.documents.models import TextChunk
from apps.documents.tests.test_embedding_generation import legacy as legacy

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


def test_failed_projection_rolls_back_chunk_publication(legacy, monkeypatch):
    from apps.documents.services import lexical_publication
    from apps.documents.tasks.chunking import _commit_chunks
    from apps.documents.tests.test_lexical_relevance import replace_chunks

    old = replace_chunks(legacy, ["source"])[0]
    type(legacy[0]).objects.filter(pk=legacy[0].pk).update(ingestion_complete=False)
    legacy[0].refresh_from_db()
    replacement = TextChunk(
        doc_id=legacy[0].id,
        content="source",
        chunk_number=0,
        start_position=0,
        end_position=6,
    )

    def fail(*args, **kwargs):
        raise ValueError("synthetic projection failure")

    monkeypatch.setattr(lexical_publication, "publish_chunk_search_vectors", fail)
    with pytest.raises(ValueError, match="projection failure"):
        _commit_chunks(
            legacy[0],
            [replacement],
            expected_source_hash=legacy[0].full_text_hash,
            using="default",
        )
    assert TextChunk.objects.get().pk == old.pk
    assert TextChunk.objects.get().search_vector_binding is None
    assert not TextChunk.objects.filter(pk=replacement.pk).exists()
    legacy[0].refresh_from_db()
    assert not legacy[0].ingestion_complete


def test_future_staged_projection_uses_validated_structure_without_pointer_activation(
    legacy,
):
    from uuid import uuid4

    from apps.documents.services.lexical_publication import publish_chunk_search_vectors
    from lib.parsers.source_contracts import (
        ExtractionV1,
        SourceElementV1,
        source_artifact_ref,
    )

    canonical = "Measured\nATP BRCA1"
    artifact = source_artifact_ref(
        canonical.encode(), member_path=None, container_sha256=None
    )
    extraction = ExtractionV1(
        "synthetic-v1",
        artifact,
        canonical,
        (
            SourceElementV1("heading", "heading", 0, 9, None, None, None, None),
            SourceElementV1(
                "body", "paragraph", 9, len(canonical), None, None, None, "heading"
            ),
        ),
        "complete",
        (),
    )
    generation = uuid4()
    TextChunk.objects.all().delete()
    chunk = TextChunk(
        doc_id=legacy[0].id,
        content=canonical[9:],
        chunk_number=0,
        start_position=9,
        end_position=len(canonical),
        representation_generation=generation,
    )
    with transaction.atomic():
        TextChunk.objects.bulk_create([chunk])
        publish_chunk_search_vectors(
            document=legacy[0],
            chunks=(chunk,),
            using="default",
            canonical_text=canonical,
            representation_key=generation,
            source_revision_sha256="a" * 64,
            extraction=extraction,
        )
    chunk.refresh_from_db()
    assert chunk.source_kind == "body"
    assert chunk.section_path == ["Measured"]
    assert chunk.search_vector_binding["representation_key"] == str(generation)
    assert chunk.search_vector_binding["source_revision_sha256"] == "a" * 64
    assert chunk.content == canonical[9:]
    legacy[0].refresh_from_db()
    assert legacy[0].full_text == "source"


def test_staged_binding_uses_post_update_owner_hash_and_separate_input_revision(legacy):
    from uuid import uuid4

    from apps.documents.services.lexical_projection import text_hash
    from apps.documents.services.lexical_publication import publish_chunk_search_vectors

    doc = legacy[0]
    original_hash = doc.full_text_hash
    canonical, generation = "Parsed canonical body", uuid4()
    with transaction.atomic():
        locked = type(doc).objects.select_for_update().get(pk=doc.pk)
        assert locked.full_text_hash == original_hash
        TextChunk.objects.filter(doc_id=doc.id).delete()
        chunk = TextChunk(
            doc_id=doc.id,
            content=canonical,
            chunk_number=0,
            start_position=0,
            end_position=len(canonical),
            representation_generation=generation,
        )
        TextChunk.objects.bulk_create([chunk])
        type(doc).objects.filter(pk=doc.pk).update(
            full_text=canonical, full_text_hash=text_hash(canonical)
        )
        locked.refresh_from_db()
        publish_chunk_search_vectors(
            document=locked,
            chunks=(chunk,),
            using="default",
            canonical_text=canonical,
            representation_key=generation,
            source_revision_sha256=original_hash,
        )
    chunk.refresh_from_db()
    assert chunk.search_vector_binding["document"]["source_hash"] == text_hash(
        canonical
    )
    assert chunk.search_vector_binding["source_revision_sha256"] == original_hash
