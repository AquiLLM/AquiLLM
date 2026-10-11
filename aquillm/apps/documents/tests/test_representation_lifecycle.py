"""Membership, predecessor and deletion races retain history without stale reads."""
import pytest

from apps.collections.models import Collection, CollectionPermission
from apps.documents.models import TextChunk, DocumentRepresentationState
from apps.documents.services.representation_activation import activate_representation, stage_representation
from apps.documents.services.representation_snapshot_build import begin_canonical_snapshot_representation, build_canonical_snapshot_representation
from apps.documents.services.representation_store import freeze_active_representations, load_chunk_source
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from .representation_support import retained as retained, upload as upload, reference
from .test_representation_edits import manual

pytestmark = pytest.mark.django_db(transaction=True)


def destination(doc):
    collection = Collection.objects.create(name="Destination")
    CollectionPermission.objects.create(user=doc.ingested_by, collection=collection, permission="EDIT")
    return collection


def staged(doc):
    lease, chunks, predecessor = begin_canonical_snapshot_representation(reference(doc), using="default")
    stage_representation(lease, chunks=chunks, using="default")
    return lease, predecessor


def test_move_preserves_history_under_current_owner_and_allows_rebuild(retained):
    doc = manual(retained)
    old = TextChunk.objects.get(doc_id=doc.id)
    before = freeze_active_representations((reference(doc),), using="default")
    lease, predecessor = staged(doc)
    target = destination(doc)
    doc.move_to(target, actor=doc.ingested_by)
    doc.refresh_from_db()
    assert TextChunk.objects.filter(pk=old.pk).exists()
    assert DocumentRepresentationState.objects.get(document_id=doc.id).collection_id == target.pk
    assert load_chunk_source(old.pk, using="default").document.collection_id == target.pk
    with pytest.raises(SourceScopeUnavailable):
        freeze_active_representations((before.selections[0].document,), using="default")
    with pytest.raises(SourceScopeUnavailable):
        freeze_active_representations((reference(doc),), using="default")
    assert activate_representation(lease, expected_active_key=predecessor, using="default") == "stale"
    assert build_canonical_snapshot_representation(reference(doc), using="default") == "activated"
    assert TextChunk.objects.filter(pk=old.pk).exists()


def test_newer_activation_fences_old_staged_completion(retained):
    doc = manual(retained)
    old_lease, predecessor = staged(doc)
    new_lease, _ = staged(doc)
    assert activate_representation(new_lease, expected_active_key=predecessor, using="default") == "activated"
    assert activate_representation(old_lease, expected_active_key=predecessor, using="default") == "stale"
    assert DocumentRepresentationState.objects.get(document_id=doc.id).active_generation_id == new_lease.generation_key


def test_deletion_before_completion_fails_closed_without_recreating_history(retained):
    doc = manual(retained)
    lease, predecessor = staged(doc)
    doc.delete()
    assert activate_representation(lease, expected_active_key=predecessor, using="default") == "stale"
    assert not TextChunk.objects.filter(doc_id=doc.id).exists()
