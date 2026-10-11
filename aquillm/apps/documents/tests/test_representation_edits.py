"""Real owner edits revoke current eligibility and retain immutable citation text."""
import pytest

from apps.documents.models import RawTextDocument, TextChunk, DocumentRepresentationState
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.services.representation_snapshot_build import (
    begin_canonical_snapshot_representation, build_canonical_snapshot_representation,
)
from apps.documents.services.representation_activation import activate_representation, stage_representation
from apps.documents.services.representation_store import freeze_active_representations, load_chunk_source
from .representation_support import retained as retained, upload as upload, reference

pytestmark = pytest.mark.django_db(transaction=True)


def manual(retained):
    original, _, _ = retained
    doc = RawTextDocument(title="Manual", full_text="first manual revision", collection=original.collection,
                          ingested_by=original.ingested_by)
    doc.save(dont_rechunk=True)
    assert build_canonical_snapshot_representation(reference(doc), using="default") == "activated"
    doc.refresh_from_db()
    return doc


def test_edit_revokes_current_then_rebuilds_without_replacing_history(retained):
    doc = manual(retained)
    old = TextChunk.objects.get(doc_id=doc.id)
    frozen = freeze_active_representations((reference(doc),), using="default")
    doc.full_text = "second manual revision"
    doc.save(dont_rechunk=True)
    with pytest.raises(SourceScopeUnavailable):
        freeze_active_representations((reference(doc),), using="default")
    assert load_chunk_source(old.pk, using="default").canonical_text == "first manual revision"
    assert build_canonical_snapshot_representation(reference(doc), using="default") == "activated"
    current = freeze_active_representations((reference(doc),), using="default")
    assert current.selections[0].generation_key != frozen.selections[0].generation_key
    assert TextChunk.objects.filter(pk=old.pk, content="first manual revision").exists()


def test_edit_after_capture_rejects_stage_without_chunks(retained):
    doc = manual(retained)
    lease, chunks, _ = begin_canonical_snapshot_representation(reference(doc), using="default")
    doc.full_text = "changed after immutable input capture"
    doc.save(dont_rechunk=True)
    with pytest.raises(SourceScopeUnavailable):
        stage_representation(lease, chunks=chunks, using="default")
    assert not TextChunk.objects.filter(representation_generation_id=lease.generation_key).exists()


def test_edit_after_stage_rolls_back_activation(retained):
    doc = manual(retained)
    lease, chunks, predecessor = begin_canonical_snapshot_representation(reference(doc), using="default")
    stage_representation(lease, chunks=chunks, using="default")
    doc.full_text = "changed after staging"
    doc.save(dont_rechunk=True)
    assert activate_representation(lease, expected_active_key=predecessor, using="default") == "stale"
    doc.refresh_from_db()
    assert doc.full_text == "changed after staging"
    assert DocumentRepresentationState.objects.get(document_id=doc.id).active_generation_id == predecessor
