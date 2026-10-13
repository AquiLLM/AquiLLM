"""Synthetic broker only; retain actual representation rebuild request/resume."""
from contextlib import contextmanager
from copy import deepcopy

from apps.documents.models import DocumentRepresentationState
from apps.knowledge_graph.models import GraphRebuildRequest
from apps.knowledge_graph.services import builds
from apps.knowledge_graph.tasks import build_document_graph_task
from apps.knowledge_graph.tests.test_mention_extraction import _persist_active_ontology


class RebuildPublications:
    def __init__(self):
        self.expected = {}
        self.emitted = {}
        self.attempts = []

    def expect(self, document):
        key = str(document.id)
        assert key not in self.expected
        self.expected[key] = document

    def capture(self, task, *, kwargs, build_kind, scope_id):
        self.attempts.append(scope_id)
        assert task is build_document_graph_task and build_kind == "document"
        assert scope_id in self.expected and scope_id not in self.emitted
        doc = self.expected[scope_id]
        doc.refresh_from_db()
        request = GraphRebuildRequest.objects.get(pk=kwargs["request_id"])
        state = DocumentRepresentationState.objects.get(
            model_label=doc._meta.label_lower, document_pkid=doc.pkid, document_id=doc.id)
        assert state.active_generation_id is not None
        expected = {
            "document_id": str(doc.id), "document_pkid": doc.pkid,
            "model_label": doc._meta.label_lower, "collection_id": doc.collection_id,
            "source_hash": doc.full_text_hash,
            "representation_binding": {"version": "graph-document-representation-v1",
                                       "generation_key": str(state.active_generation_id)},
        }
        assert request.scope_type == "document" and request.scope_id == scope_id
        assert request.requested_documents == [expected]
        assert request.status == "running" and request.document_count == 1
        assert request.document_publish_cursor == 0
        assert request.document_publication_state == "pending"
        assert request.evaluation_only is False
        assert kwargs == {
            "document_id": str(doc.id), "expected_source_hash": doc.full_text_hash,
            "document_build_key": builds.derive_current_document_build_key(doc.id, doc.full_text_hash),
            "request_id": str(request.pk), "eval_only": False,
        }
        self.emitted[scope_id] = (request.pk, deepcopy(expected), deepcopy(kwargs))

    def assert_published(self, document):
        # Runs outside the production robust on_commit callback: failures cannot hide.
        assert self.attempts.count(str(document.id)) == 1
        request_id, expected, payload = self.emitted[str(document.id)]
        request = GraphRebuildRequest.objects.get(pk=request_id)
        assert request.requested_documents == [expected]
        assert request.document_publish_cursor == request.document_count == 1
        assert request.document_publication_state == "published"
        assert request.error_code == "" and request.status == "running"
        assert request.completed_document_count == 0 and request.activated_artifact_pk is None
        assert payload["request_id"] == str(request.pk)


@contextmanager
def observed_rebuild_publications(monkeypatch):
    # Same broker seam as test_representation_graph_pipeline; no task execution.
    _persist_active_ontology()
    observed = RebuildPublications()
    with monkeypatch.context() as broker:
        broker.setenv("KG_BUILD_ENABLED", "true")
        broker.setenv("KG_GRAPH_REMEDIATION_MODE", "v2")
        broker.setattr(builds, "_publish_graph_task", observed.capture)
        yield observed
        assert set(observed.expected) == set(observed.emitted)
        assert len(observed.attempts) == len(observed.expected)
