"""Legacy-null eligibility and payload sharing use genuine persisted source owners."""
import pytest
from django.db import connections
from django.test.utils import CaptureQueriesContext

from apps.knowledge_graph.models import CollectionGraphProjection, GraphArtifact
from apps.knowledge_graph.projection.worker import project_generation
from apps.knowledge_graph.services import builds
from apps.knowledge_graph.tasks import build_document_graph_task, refresh_collection_graph_task
from tests.integration.projection_boundary_support import (
    boundary_projection, graph_configuration, graph_source, projection_roles,
    source_call, state_repository, upload,
)

pytestmark = pytest.mark.django_db(
    transaction=True, databases=["default", "projection_source", "projection_state"])


@pytest.mark.parametrize("mode", ["legacy", "v2"])
def test_real_legacy_source_producer_keeps_exact_null_selection(graph_source, projection_roles, monkeypatch, mode):
    source = graph_source
    monkeypatch.setenv("KG_GRAPH_REMEDIATION_MODE", mode)
    from apps.documents.models import DocumentRepresentationState, TextChunk
    # Historical pre-generation retained input, not today's chunk worker output.
    # Its text and exact owner come from the real parsed source; no active
    # generation is removed or rewritten to manufacture a legacy selection.
    assert not DocumentRepresentationState.objects.filter(document_id=source.document.id).exists()
    assert not source.document.chunks.exists()
    text = source.document.full_text
    TextChunk.objects.bulk_create([TextChunk(
        doc_id=source.document.id, content=text, start_position=0,
        end_position=len(text), chunk_number=0,
    )])
    source.document.ingestion_complete = True
    source.document.save(dont_rechunk=True, update_fields=["ingestion_complete"])
    assert source.document.chunks.exists()
    assert not source.document.chunks.exclude(representation_generation_id=None).exists()
    builds.enqueue_document_build(source.document.id, source.document.full_text_hash)
    task, payload = next((task, payload) for task, payload in source.emitted
                         if task is build_document_graph_task)
    artifact = GraphArtifact.objects.get(pk=task.run(**payload))
    assert artifact.status == "active"
    task, payload = next((task, payload) for task, payload in reversed(source.emitted)
                         if task is refresh_collection_graph_task)
    collection_artifact = GraphArtifact.objects.get(pk=task.run(**payload))
    row = CollectionGraphProjection.objects.get(artifact=collection_artifact, state="pending")
    if mode == "legacy":
        assert row.representation_snapshot is None
    else:
        assert row.representation_snapshot["selections"][0]["generation_key"] is None
    outcome = project_generation(projection_id=row.pk, lease_owner="boundary-worker")
    assert outcome.ready, outcome.failure_code
    row.refresh_from_db()
    assert source_call(row, purpose="audit", lease_owner=None)
    _, observed = source_call(row, purpose="audit", lease_owner=None, coordinates=True)
    assert observed == list(source.document.chunks.order_by("pk").values_list("pk", "doc_id", "chunk_number"))
    # The exact old null selection must close after the real first activation.
    source.activate_only("Orion uses MMLU. First bound generation.")
    state = DocumentRepresentationState.objects.get(document_id=source.document.id)
    assert state.legacy_closed and state.active_generation_id is not None
    assert source_call(row, purpose="audit", lease_owner=None) is False


def test_private_reference_payload_is_fetched_and_parsed_once(boundary_projection, monkeypatch):
    from apps.knowledge_graph.projection import representation_binding
    from apps.knowledge_graph.projection.chunk_reference_store import validate_current_chunk_rows
    _, row = boundary_projection
    assert project_generation(projection_id=row.pk, lease_owner="boundary-worker").ready
    repository = state_repository()
    with CaptureQueriesContext(connections["projection_source"]) as reads:
        references = repository.load(projection_id=row.pk)
    assert references
    binding_reads = [q["sql"] for q in reads if '"representation_snapshot"' in q["sql"]]
    assert len(binding_reads) == 1
    assert len({id(ref.projection) for ref in references}) == 1
    seen = []
    original = representation_binding.validate_record_binding

    def validate(record):
        seen.append(record)
        return original(record)

    monkeypatch.setattr(representation_binding, "validate_record_binding", validate)
    # Repeat the real rows only to exercise batch parsing; no invented source authority.
    with CaptureQueriesContext(connections["projection_source"]) as later:
        validate_current_chunk_rows(references * 5)
    assert len(later) == 0
    assert seen == [references[0].projection]
