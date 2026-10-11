"""Real I5 source and guarded G5/collection task producer."""
from dataclasses import dataclass, field

import pytest

from apps.documents.models import RawTextDocument
from apps.documents.tests.representation_support import activate
from apps.ingestion.tests.output_support import upload as upload
from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun, GraphRebuildRequest
from apps.knowledge_graph.services import builds
from apps.knowledge_graph.tasks import build_document_graph_task, refresh_collection_graph_task
from apps.knowledge_graph.tests.test_mention_extraction import _persist_active_ontology, _result
from aquillm.task_ingest_uploaded import run_ingest_uploaded_file


class SourceValidBackend:
    """Only the external extraction boundary is synthetic; spans are source exact."""

    def __init__(self):
        self.calls = []

    def extract_batch(self, texts, *, ontology):
        self.calls.append((tuple(texts), ontology.checksum))
        return tuple(_result(model_start=text.index("Orion"),
                             dataset_start=text.index("MMLU"), confidence=0.95)
                     for text in texts)


@dataclass
class PersistedGraphSource:
    document: object
    output: object
    backend: SourceValidBackend
    emitted: list = field(default_factory=list)

    def capture(self, task, **kwargs):
        self.emitted.append((task, dict(kwargs["kwargs"])))

    def activate_and_build_document(self, text):
        return self.build_document_from(self.activate_only(text))

    def activate_only(self, text):
        before = len(self.emitted)
        assert activate(self.document, self.output, text) == "activated"
        self.document.refresh_from_db()
        return before

    def build_document_from(self, before):
        pending = [(task, payload) for task, payload in self.emitted[before:]
                   if task is build_document_graph_task]
        assert len(pending) == 1, "real publication must emit one exact guarded task"
        task, payload = pending[0]
        request = GraphRebuildRequest.objects.get(pk=payload["request_id"])
        assert request.document_publication_state == "published"
        assert request.requested_documents[0]["representation_binding"]["generation_key"]
        artifact_id = task.run(**payload)
        artifact = GraphArtifact.objects.get(pk=artifact_id)
        run = GraphBuildRun.objects.get(artifact=artifact)
        request.refresh_from_db()
        assert artifact.status == "active" and run.status == "succeeded"
        assert artifact.source_hash == self.document.full_text_hash
        assert artifact.representation_snapshot is not None
        assert request.status == "succeeded"
        assert request.activated_artifact_pk == artifact.pk
        assert self.backend.calls
        return artifact

    def build_emitted_collection(self):
        pending = [(task, payload) for task, payload in self.emitted
                   if task is refresh_collection_graph_task
                   and payload["collection_id"] == self.document.collection_id]
        if not pending:
            # Surface the real identity error swallowed by the async publisher.
            builds._collection_context(self.document.collection_id)
        assert pending, "document activation must publish the real collection refresh"
        task, payload = pending[-1]
        artifact_id = task.run(**payload)
        artifact = GraphArtifact.objects.get(pk=artifact_id)
        assert artifact.scope_type == "collection" and artifact.status == "active"
        assert artifact.collection_scope_id == self.document.collection_id
        assert artifact.representation_snapshot is not None
        assert GraphBuildRun.objects.get(artifact=artifact).status == "succeeded"
        return artifact


@pytest.fixture
def graph_source(upload, graph_configuration, monkeypatch, settings):
    from apps.knowledge_graph.extraction import pipeline

    _persist_active_ontology()
    monkeypatch.setenv("KG_BUILD_ENABLED", "true")
    monkeypatch.setenv("KG_GRAPH_REMEDIATION_MODE", "v2")
    settings.KG_EXTRACTION_QUEUE_VALID = True
    backend = SourceValidBackend()
    monkeypatch.setattr(pipeline, "_build_backend", lambda config: backend)
    item = upload((("source.txt", b"Orion uses MMLU. Original observation."),))
    run_ingest_uploaded_file(item.pk)
    output = item.outputs.get(status="success")
    source = PersistedGraphSource(
        RawTextDocument.objects.get(pk=output.document_pkid), output, backend
    )
    monkeypatch.setattr(builds, "_publish_graph_task", source.capture)
    return source
