"""Legacy source-body consumption revalidates the exact captured graph capsule."""
from types import SimpleNamespace
import pytest

from apps.documents.models import TextChunk
from apps.documents.services.chunk_source_materialization import materialize_graph_rows
from apps.documents.services.source_loading import source_runtime_scope
from apps.knowledge_graph.tests.test_vocabulary_ready_authority import prepared


@pytest.mark.django_db(transaction=True)
def test_legacy_binding_and_final_materialization_reject_detached_policy(monkeypatch):
    from apps.knowledge_graph.models import CollectionArtifactInput, GraphArtifact
    from apps.knowledge_graph.resolution.vocabulary_authority import bind_legacy
    fixture, runtime, pin = prepared(monkeypatch, bound=True)
    projection = fixture.projections["A"]
    artifact_id = projection.artifact_id
    fields = ("id", "scope_type", "scope_id", "collection_scope_id", "resolver_version",
              "resolution_config_checksum", "metadata")
    collections = tuple(GraphArtifact.objects.filter(pk=artifact_id).values(*fields))
    manifests = tuple(CollectionArtifactInput.objects.filter(artifact_id=artifact_id).values(
        "document_artifact_id", "collection_id",
        *("document_artifact__" + field for field in fields)))
    request = SimpleNamespace(allowed_doc_ids=(fixture.documents["A"].id,), allowed_collection_ids=(7,))
    chunk = fixture.chunks["A"]
    with source_runtime_scope(runtime):
        bind_legacy(request, collections, manifests)
        assert materialize_graph_rows(TextChunk, (chunk.pk,), request.allowed_doc_ids) == ((chunk,), 0)
        pin.delete()
        assert materialize_graph_rows(TextChunk, (chunk.pk,), request.allowed_doc_ids) == ((), 1)


@pytest.mark.django_db(transaction=True)
def test_legacy_materializer_rejects_missing_original_runtime(monkeypatch):
    fixture, _, _ = prepared(monkeypatch, bound=True)
    chunk = fixture.chunks["A"]
    assert materialize_graph_rows(TextChunk, (chunk.pk,), (chunk.doc_id,)) == ((), 1)


@pytest.mark.django_db(transaction=True)
def test_legacy_final_artifact_database_failure_is_unavailable(monkeypatch):
    from django.db import OperationalError
    from django.db.models.query import QuerySet
    from apps.knowledge_graph.models import CollectionArtifactInput, GraphArtifact
    from apps.knowledge_graph.resolution.vocabulary_authority import bind_legacy
    fixture, runtime, _ = prepared(monkeypatch, bound=True)
    aid = fixture.projections["A"].artifact_id
    fields = ("id", "scope_type", "scope_id", "collection_scope_id", "resolver_version",
              "resolution_config_checksum", "metadata")
    collections = tuple(GraphArtifact.objects.filter(pk=aid).values(*fields))
    manifests = tuple(CollectionArtifactInput.objects.filter(artifact_id=aid).values(
        "document_artifact_id", "collection_id",
        *("document_artifact__" + field for field in fields)))
    request = SimpleNamespace(allowed_doc_ids=(fixture.documents["A"].id,), allowed_collection_ids=(7,))
    chunk = fixture.chunks["A"]
    original = QuerySet.values
    def fail_artifact_read(queryset, *args, **kwargs):
        if queryset.model is GraphArtifact:
            raise OperationalError("synthetic unavailable artifact query")
        return original(queryset, *args, **kwargs)
    with source_runtime_scope(runtime):
        bind_legacy(request, collections, manifests)
        monkeypatch.setattr(QuerySet, "values", fail_artifact_read)
        assert materialize_graph_rows(TextChunk, (chunk.pk,), request.allowed_doc_ids) == ((), 1)

