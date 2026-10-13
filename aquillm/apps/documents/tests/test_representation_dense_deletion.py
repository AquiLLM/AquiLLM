"""All retained representation and dense sidecar rows share the lifecycle cap."""
from unittest.mock import patch
import pytest

from apps.documents.models import TextChunk
from apps.documents.models.embedding_index import DocumentEmbeddingState, EmbeddingGeneration, ChunkEmbedding
from apps.documents.services import embedding_index_store as store
from apps.documents.services.representation_store import freeze_active_representations
from apps.documents.tasks.embedding_index import rebuild_document_embeddings, build_embedding_generation
from lib.embeddings.tests.test_document_provider import transport as transport
from .representation_support import retained as retained, upload as upload, activate, reference

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.mark.parametrize("queryset", [False, True])
def test_dense_history_counts_once_and_refuses_overcap_before_mutation(retained, transport, monkeypatch, queryset):
    from apps.knowledge_graph.graph import invalidation
    doc, output, old = retained
    assert activate(doc, output) == "activated"
    doc.refresh_from_db()
    selection = freeze_active_representations((reference(doc),), using="default").selections[0]
    store.persist_embedding_space(transport[0], using="default")
    with patch.object(build_embedding_generation, "delay"):
        key = rebuild_document_embeddings(selection, space_key=transport[0].space_key, using="default")
    assert build_embedding_generation.run(key, using="default") == "activated"
    collector = invalidation._BoundedDeleteCollector(using="default", origin=doc)
    collector.collect([doc])
    for model in (DocumentEmbeddingState, EmbeddingGeneration, ChunkEmbedding):
        assert model.objects.exists()
        assert set(model.objects.all()) <= collector.data[model]
    count = collector._bounded_source_count
    collector.collect([doc])
    assert collector._bounded_source_count == count
    monkeypatch.setattr(invalidation, "_MAX_LIFECYCLE_ROWS", count - 1)
    with pytest.raises(RuntimeError, match="bounded"):
        (type(doc).objects.filter(pk=doc.pk) if queryset else doc).delete()
    assert TextChunk.objects.filter(pk=old.pk).exists() and ChunkEmbedding.objects.exists()
    monkeypatch.setattr(invalidation, "_MAX_LIFECYCLE_ROWS", count)
    monkeypatch.setattr(invalidation._bounded_values, "__kwdefaults__", {"maximum": count})
    (type(doc).objects.filter(pk=doc.pk) if queryset else doc).delete()
    assert not DocumentEmbeddingState.objects.exists() and not TextChunk.objects.filter(pk=old.pk).exists()
