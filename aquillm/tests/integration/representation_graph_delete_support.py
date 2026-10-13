"""Real representation/graph producers; no fake deletion authority or callbacks."""
from dataclasses import dataclass
import hashlib
import json

import pytest

from apps.documents.models import (
    DocumentRepresentationGeneration, DocumentRepresentationState,
    LegacyDocumentSourceSnapshot, TextChunk,
)
from apps.knowledge_graph.models import (
    CollectionGraphProjection, CollectionRelationEvidence, GraphArtifact,
    GraphBuildRun, GraphRebuildRequest,
)
from tests.integration.persisted_graph_projection_support import graph_configuration
from tests.integration.persisted_graph_source_support import graph_source, upload

HISTORY = (DocumentRepresentationState, DocumentRepresentationGeneration,
           LegacyDocumentSourceSnapshot, TextChunk)


def row_digest(model, keys):
    fields = [field.attname for field in model._meta.concrete_fields
              if field.get_internal_type() != "VectorField"]
    rows = list(model.objects.filter(pk__in=keys).order_by("pk").values(*fields))
    return hashlib.sha256(json.dumps(rows, sort_keys=True, default=str).encode()).hexdigest()


@dataclass
class DeleteCase:
    source: object
    owner_pk: int
    document_id: object
    artifact_ids: tuple
    audit_ids: tuple
    request_id: object
    activated_artifact_pk: int
    projection_ids: tuple
    membership: dict

    @property
    def document(self):
        return self.source.document

    def snapshot(self):
        models = {type(self.document): (self.owner_pk,), **self.membership,
                  GraphArtifact: self.artifact_ids, GraphBuildRun: self.audit_ids,
                  GraphRebuildRequest: (self.request_id,),
                  CollectionGraphProjection: self.projection_ids}
        return {model: row_digest(model, keys) for model, keys in models.items()}

    def assert_deleted(self):
        assert not type(self.document).objects.filter(pk=self.owner_pk).exists()
        for model, keys in self.membership.items():
            assert not model.objects.filter(pk__in=keys).exists(), model.__name__
        assert not GraphArtifact.objects.filter(pk__in=self.artifact_ids).exists()
        assert not CollectionRelationEvidence.objects.filter(artifact_id__in=self.artifact_ids).exists()
        assert not CollectionGraphProjection.objects.filter(
            pk__in=self.projection_ids, state__in=("pending", "building", "ready")).exists()
        for audit in GraphBuildRun.objects.filter(pk__in=self.audit_ids):
            assert audit.artifact_id is None and audit.status == "cancelled"
            assert audit.stage == "stale"
        assert GraphBuildRun.objects.filter(pk__in=self.audit_ids).count() == len(self.audit_ids)
        request = GraphRebuildRequest.objects.get(pk=self.request_id)
        assert request.status == "succeeded"
        assert request.activated_artifact_pk == self.activated_artifact_pk
        request._validate_success_activation()


@pytest.fixture
def graph_delete_case(graph_source):
    source = graph_source
    source.activate_and_build_document("Orion uses MMLU. First retained generation.")
    current = source.activate_and_build_document("Orion uses MMLU. Current retained generation.")
    collection = source.build_emitted_collection()
    state = DocumentRepresentationState.objects.get(
        model_label=source.document._meta.label_lower,
        document_pkid=source.document.pkid, document_id=source.document.id)
    generations = tuple(state.generations.values_list("pk", flat=True))
    assert len(generations) == 2 and state.active_generation_id in generations
    chunks = tuple(TextChunk.objects.filter(doc_id=source.document.id).values_list("pk", flat=True))
    assert len(chunks) >= 2
    assert TextChunk.objects.filter(pk__in=chunks).exclude(
        representation_generation_id=state.active_generation_id).exists()
    artifact_ids = (current.pk, collection.pk)
    assert CollectionRelationEvidence.objects.filter(artifact=collection).exists()
    audit_ids = tuple(GraphBuildRun.objects.filter(artifact_id__in=artifact_ids).values_list("pk", flat=True))
    assert len(audit_ids) == 2
    request = GraphRebuildRequest.objects.get(activated_artifact_pk=current.pk)
    assert request.status == "succeeded"
    projections = tuple(CollectionGraphProjection.objects.filter(artifact=collection).values_list("pk", flat=True))
    assert projections
    return DeleteCase(source, source.document.pk, source.document.id, artifact_ids,
        audit_ids, request.pk, current.pk, projections,
        {DocumentRepresentationState: (state.pk,), DocumentRepresentationGeneration: generations,
         LegacyDocumentSourceSnapshot: tuple(state.source_snapshots.values_list("pk", flat=True)),
         TextChunk: chunks})
