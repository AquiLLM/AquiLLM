"""Real persisted nonnull representation through guarded graph task and completion."""
import pytest

from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun, GraphRebuildRequest
from apps.knowledge_graph.services import builds
from apps.knowledge_graph.tasks import build_document_graph_task
from apps.knowledge_graph.tests.test_mention_extraction import _Backend, _result, _persist_active_ontology
from .representation_support import retained as retained, upload as upload, activate

pytestmark = pytest.mark.django_db(transaction=True)


def test_legacy_graph_keeps_null_behavior_and_refuses_retained_history(retained, monkeypatch):
    from apps.documents.services.representation_contracts import SourceScopeUnavailable
    from apps.knowledge_graph.extraction.pipeline import _ordered_chunks
    doc, output, old = retained
    monkeypatch.setenv("KG_GRAPH_REMEDIATION_MODE", "legacy")
    assert [chunk.pk for chunk in _ordered_chunks(doc.id)] == [old.pk]
    assert activate(doc, output) == "activated"
    with pytest.raises(SourceScopeUnavailable):
        _ordered_chunks(doc.id)


def test_canonical_change_publishes_and_completes_real_guarded_worker(retained, monkeypatch, settings):
    doc, output, old = retained
    _persist_active_ontology()
    monkeypatch.setenv("KG_BUILD_ENABLED", "true")
    monkeypatch.setenv("KG_GRAPH_REMEDIATION_MODE", "v2")
    settings.KG_EXTRACTION_QUEUE_VALID = True
    from apps.knowledge_graph.extraction import pipeline
    backend = _Backend((_result(model_start=0, dataset_start=11, confidence=0.95),))
    monkeypatch.setattr(pipeline, "_build_backend", lambda settings: backend)
    published = []
    monkeypatch.setattr(builds, "_publish_graph_task", lambda task, **kw: published.append((task, kw["kwargs"])))
    old_hash = doc.full_text_hash
    assert activate(doc, output, "Orion uses MMLU") == "activated"
    doc.refresh_from_db()
    request = GraphRebuildRequest.objects.get(scope_type="document", scope_id=str(doc.id))
    if request.document_publication_state != "published":
        # Surface the underlying guarded publisher failure, without replacing it.
        builds.enqueue_document_build(doc.id, doc.full_text_hash, request_id=request.pk,
                                      representation_request=request.requested_documents[0])
    assert request.document_publication_state == "published"
    target = request.requested_documents[0]
    assert target["source_hash"] == doc.full_text_hash != old_hash
    assert target["representation_binding"]["generation_key"]
    assert len(published) == 1
    task, payload = published[0]
    assert task is build_document_graph_task
    result = build_document_graph_task.run(**payload)
    artifact = GraphArtifact.objects.get(pk=result)
    run = GraphBuildRun.objects.get(artifact=artifact)
    assert artifact.status == "active" and run.status == "succeeded"
    assert artifact.representation_snapshot is not None
    assert artifact.source_hash == doc.full_text_hash
    request.refresh_from_db()
    assert request.status == "succeeded" and request.completed_document_count == 1
    assert request.activated_artifact_pk == artifact.pk
    assert backend.calls and old.pk not in artifact.entity_mentions.values_list("chunk_id", flat=True)
