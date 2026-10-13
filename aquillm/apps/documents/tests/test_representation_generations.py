"""Generation publication must preserve citations and use real persisted ownership."""
import pytest
from django.db import DatabaseError, transaction

from apps.documents.models import TextChunk
from apps.documents.services import representation_store
from .representation_support import retained as retained, upload as upload, activate, reference

pytestmark = pytest.mark.django_db(transaction=True)


def test_parser_change_retains_old_chunk_and_selects_only_new_generation(retained):
    doc, output, old = retained
    old_hash = doc.full_text_hash
    activate(doc, output)
    doc.refresh_from_db()
    old.refresh_from_db()
    assert (old.content, old.start_position, old.end_position) == ("old table 10", 0, 12)
    assert doc.full_text == "corrected table 100"
    assert doc.full_text_hash != old_hash
    output.refresh_from_db()
    assert output.document_source_hash == old_hash
    snapshot = representation_store.freeze_active_representations((reference(doc),), using="default")
    selected = snapshot.selections[0]
    assert selected.generation_key is not None
    assert selected.source_revision_sha256 is not None
    rows = list(representation_store.active_chunk_scope(TextChunk.objects.all(), snapshot=snapshot))
    assert old.pk not in [row.pk for row in rows]
    assert "".join(row.content for row in rows) == doc.full_text
    assert rows[0].search_vector_binding["document"]["source_hash"] == doc.full_text_hash
    assert rows[0].search_vector_binding["source_revision_sha256"] == selected.source_revision_sha256


def test_published_chunk_cannot_be_mutated_or_deleted_while_owner_lives(retained):
    doc, output, _ = retained
    activate(doc, output)
    row = TextChunk.objects.exclude(representation_generation__isnull=True).get()
    with pytest.raises(DatabaseError):
        with transaction.atomic():
            TextChunk.objects.filter(pk=row.pk).update(content="forged")
    with pytest.raises(DatabaseError):
        with transaction.atomic():
            TextChunk.objects.filter(pk=row.pk).delete()
    assert TextChunk.objects.filter(pk=row.pk).exists()


def test_document_delete_collects_all_generations_and_retained_chunks(retained):
    from apps.documents.models import DocumentRepresentationState
    doc, output, _ = retained
    activate(doc, output)
    doc.refresh_from_db()
    doc.delete()
    assert not TextChunk.objects.exists()
    assert not DocumentRepresentationState.objects.exists()
    assert type(output).objects.filter(pk=output.pk).exists()
