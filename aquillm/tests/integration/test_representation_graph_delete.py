"""Owner deletion retains graph ordering, history bounds, rollback and typed scope."""
import pytest
from django.db import DatabaseError, transaction
from django.test import RequestFactory

from apps.documents.models import DocumentFigure, PDFDocument, TextChunk
from apps.documents.services.representation_activation import build_document_representation
from apps.documents.services.representation_caption import caption_chunker_key
from apps.documents.tests.representation_support import reference
from apps.documents.tests.test_representation_media import figure
from apps.documents.views.api import delete_document
from apps.knowledge_graph.graph import invalidation
from tests.integration.representation_graph_delete_support import (
    graph_configuration, graph_delete_case, graph_source, row_digest, upload,
)

pytestmark = pytest.mark.django_db(transaction=True)


def remove(case, path):
    doc = case.document
    if path == "instance":
        doc.delete()
    elif path == "queryset":
        type(doc).objects.filter(pk=case.owner_pk, id=case.document_id).delete()
    elif path == "collection":
        doc.collection.delete()
    else:
        request = RequestFactory().delete("/unused/")
        request.user = doc.ingested_by
        response = delete_document(request, doc.id)
        assert response.status_code == 200, response.content


@pytest.mark.parametrize("path", ["instance", "queryset", "collection", "api"])
def test_real_owner_delete_cleans_graph_and_retained_history(graph_delete_case, path):
    case = graph_delete_case
    remove(case, path)
    case.assert_deleted()


@pytest.mark.parametrize("path", ["instance", "queryset", "collection"])
def test_outer_rollback_restores_owner_history_graph_and_audit(graph_delete_case, path):
    case = graph_delete_case
    before = case.snapshot()
    with pytest.raises(RuntimeError, match="rollback sentinel"):
        with transaction.atomic():
            remove(case, path)
            case.assert_deleted()
            raise RuntimeError("rollback sentinel")
    assert case.snapshot() == before


def test_graph_owner_history_is_counted_once_before_overflow(graph_delete_case, monkeypatch):
    case = graph_delete_case
    before = case.snapshot()
    collector = invalidation._BoundedDeleteCollector(using="default", origin=case.document)
    collector.collect([case.document])
    for model, keys in case.membership.items():
        assert set(keys) <= {row.pk for row in collector.data[model]}
        assert not any(query.model is model for query in collector.fast_deletes)
    count = collector._bounded_source_count
    assert count == sum(len(rows) for rows in collector.data.values())
    collector.collect([case.document])
    assert collector._bounded_source_count == count
    assert case.snapshot() == before
    monkeypatch.setattr(invalidation, "_MAX_LIFECYCLE_ROWS", count - 1)
    with pytest.raises(RuntimeError, match="bounded"):
        case.document.delete()
    assert case.snapshot() == before


def test_cross_model_uuid_collision_cannot_borrow_graph_owner_cleanup(graph_delete_case):
    case = graph_delete_case
    doc = case.document
    collision = PDFDocument(id=doc.id, title="typed collision", collection=doc.collection,
        ingested_by=doc.ingested_by, full_text=doc.full_text, full_text_hash=doc.full_text_hash)
    PDFDocument.objects.bulk_create([collision])
    collision_pk = collision.pk
    before = case.snapshot()
    with pytest.raises((RuntimeError, ValueError, DatabaseError)):
        with transaction.atomic():
            collision.delete()
    assert PDFDocument.objects.filter(pk=collision_pk).exists()
    assert case.snapshot() == before


def test_real_figure_parent_delete_preserves_unrelated_graph_owner(graph_delete_case, figure):
    # Same collection, separate real I3 parent. This is parent/figure lifecycle
    # coverage, not a claim that a graph was extracted from image pixels.
    case = graph_delete_case
    doc, output, legacy = figure
    parent = doc.parent_document
    assert parent.id != case.document_id and parent.collection_id == case.document.collection_id
    assert build_document_representation(reference(doc),
        extraction_revision_id=output.extraction_revision_id,
        chunker_key=caption_chunker_key(), using="default") == "activated"
    state = doc.representation_states.get()
    generation_ids = tuple(state.generations.values_list("pk", flat=True))
    snapshot_ids = tuple(state.source_snapshots.values_list("pk", flat=True))
    chunk_ids = tuple(TextChunk.objects.filter(doc_id=doc.id).values_list("pk", flat=True))
    assert generation_ids and snapshot_ids and legacy.pk in chunk_ids
    from apps.documents.models import (
        DocumentRepresentationState, DocumentRepresentationGeneration, LegacyDocumentSourceSnapshot,
    )
    other = {model: row_digest(model, keys) for model, keys in case.membership.items()}
    parent.delete()
    assert not DocumentFigure.objects.filter(pk=doc.pk).exists()
    assert not DocumentRepresentationState.objects.filter(pk=state.pk).exists()
    assert not DocumentRepresentationGeneration.objects.filter(pk__in=generation_ids).exists()
    assert not LegacyDocumentSourceSnapshot.objects.filter(pk__in=snapshot_ids).exists()
    assert not TextChunk.objects.filter(pk__in=chunk_ids).exists()
    assert type(case.document).objects.filter(pk=case.owner_pk).exists()
    assert {model: row_digest(model, keys) for model, keys in case.membership.items()} == other
