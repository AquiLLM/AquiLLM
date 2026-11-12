"""Same-owner adoption must never retroactively certify original chunk bytes.

The real synthetic parser, immutable revisions, SQL guards and publication APIs
run here. Background build dispatch is held; this is not a readiness/quality test.
"""
import pytest

from apps.collections.models import CollectionPermission
from apps.documents.models import DocumentRepresentationGeneration, TextChunk
from apps.documents.services import lexical_publication, representation_activation
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.services.representation_store import active_chunk_scope, freeze_active_representations
from apps.documents.views.api import chunk_detail
from apps.ingestion.models import ExtractionRevision, IngestionOutput
from apps.ingestion.services.current_pdf_adoption import adopt_current_pdf
from apps.ingestion.services.output_status import exact_document
from apps.ingestion.tests.current_pdf_adoption_support import (
    CANONICAL, owner_snapshot, pdf_adoption_case as pdf_adoption_case, reference, s3 as s3,
)
from apps.knowledge_graph.projection.representation_inputs import validate_artifact_selection
from apps.knowledge_graph.services import builds
from .current_adoption_transition_support import (
    NATIVE_CANONICAL, assert_builds_held, assert_citation, assert_current_generation,
    assert_original_history, citation_request, native_revision, stage_native,
)
from .representation_rollback_graph_support import assert_graph_superseded, seed_graph
from .representation_rollback_support import active

# The coordinator forbids an enclosing transaction and must commit before build.
pytestmark = pytest.mark.django_db(transaction=True)


def test_adopted_pdf_native_transition_preserves_history_and_rebuilds_rollback(
    pdf_adoption_case, monkeypatch, tmp_path, settings,
):
    case, doc = pdf_adoption_case, pdf_adoption_case.document
    settings.APPLICATION_MAINTENANCE_SCHEDULER_ENABLED = False
    settings.KG_MAINTENANCE_SCHEDULER_ENABLED = False
    monkeypatch.setenv("KG_BUILD_ENABLED", "0")
    monkeypatch.setenv("KG_GRAPH_REMEDIATION_MODE", "v2")
    # Preserve real request insertion/invalidation; hold only its on-commit worker dispatch.
    monkeypatch.setattr(builds, "resume_rebuild_request", lambda request_id: None)
    before = owner_snapshot(doc)
    original = TextChunk.objects.filter(doc_id=doc.id).values().get()
    result = adopt_current_pdf(case.ref, actor_id=case.actor.pk, using="default")
    assert result.status == "success"
    assert owner_snapshot(doc) == before
    output = IngestionOutput.objects.get(output_key=result.output_key)
    frozen_output = IngestionOutput.objects.filter(pk=output.pk).values().get()
    frozen_revision = ExtractionRevision.objects.filter(pk=output.extraction_revision_id).values().get()
    assert output.provenance_kind == "current_adoption"
    revision_b, artifact = native_revision(case, output, tmp_path, monkeypatch)

    lease, chunker_key = stage_native(doc, revision_b)
    generation_b = DocumentRepresentationGeneration.objects.get(pk=lease.generation_key)
    assert generation_b.status == "staged" and generation_b.predecessor_generation_key is None
    assert generation_b.origin_output_id == output.pk
    assert generation_b.owner.active_generation_id is None
    doc.refresh_from_db()
    assert doc.full_text == CANONICAL and doc.full_text_hash == case.ref.source_hash
    historical_snapshot = assert_original_history(original)
    staged = owner_snapshot(doc)
    frozen = freeze_active_representations((reference(doc),), using="default")
    assert set(active_chunk_scope(TextChunk.objects.all(), snapshot=frozen).values_list("pk", flat=True)) == {
        original["id"]}

    def fail_lexical(**kwargs):
        raise RuntimeError("injected native activation lexical failure")

    with monkeypatch.context() as failure:
        failure.setattr(lexical_publication, "publish_chunk_search_vectors", fail_lexical)
        with pytest.raises(RuntimeError, match="injected native activation lexical failure"):
            representation_activation.activate_representation(lease, expected_active_key=None, using="default")
    assert owner_snapshot(doc) == staged
    assert assert_original_history(original) == historical_snapshot

    assert representation_activation.activate_representation(
        lease, expected_active_key=None, using="default") == "activated"
    generation_b.refresh_from_db()
    assert generation_b.status == "active"
    snapshot_b = assert_current_generation(doc, generation_b, NATIVE_CANONICAL)
    # The origin keeps its original hash; native activation cannot rewrite history.
    assert exact_document(output, using="default") is None
    assert IngestionOutput.objects.filter(pk=output.pk).values().get() == frozen_output
    assert_builds_held(doc, generation_b, snapshot_b)
    assert_citation(case, original["id"], CANONICAL, None)
    b_rows = list(TextChunk.objects.filter(representation_generation=generation_b).order_by("pk").values())
    b_chunk_id = b_rows[0]["id"]
    assert_citation(case, b_chunk_id, NATIVE_CANONICAL, artifact)
    case.grant.delete()
    request = citation_request(case)
    assert chunk_detail(request, original["id"]).status_code == 403
    assert chunk_detail(request, b_chunk_id).status_code == 403
    CollectionPermission.objects.create(user=case.actor, collection=case.collection, permission="EDIT")

    # Synthetic graph metadata verifies supersession, not semantic graph readiness.
    graph, graph_before = seed_graph(doc)
    frozen_b = DocumentRepresentationGeneration.objects.filter(pk=generation_b.pk).values().get()
    assert representation_activation.build_document_representation(reference(doc),
        extraction_revision_id=output.extraction_revision_id, chunker_key=chunker_key,
        using="default") == "activated"
    a2 = active(doc)
    assert a2.pk != generation_b.pk and a2.predecessor_generation_key == generation_b.pk
    assert a2.origin_output_id == output.pk and a2.extraction_revision_id == output.extraction_revision_id
    assert a2.parser_key == "current-validation:legacy-upload-PDF-current-validation-v1"
    assert a2.extraction_quality == "partial" and a2.issue_codes == ["structure_unavailable"]
    assert DocumentRepresentationGeneration.objects.count() == 2
    snapshot_a2 = assert_current_generation(doc, a2, CANONICAL)
    assert exact_document(output, using="default").pk == doc.pk
    generation_b.refresh_from_db()
    assert generation_b.status == "retired" and generation_b.retired_at is not None
    retired_b = DocumentRepresentationGeneration.objects.filter(pk=generation_b.pk).values().get()
    assert {k: v for k, v in retired_b.items() if k not in {"status", "retired_at"}} == {
        k: v for k, v in frozen_b.items() if k not in {"status", "retired_at"}}
    assert list(TextChunk.objects.filter(representation_generation=generation_b).order_by("pk").values()) == b_rows
    assert_graph_superseded(graph, graph_before)
    with pytest.raises(SourceScopeUnavailable, match="source_scope_unavailable"):
        validate_artifact_selection(graph["document_artifact"], snapshot_a2.selections[0], using="default")
    assert_builds_held(doc, a2, snapshot_a2)
    assert assert_original_history(original) == historical_snapshot
    assert_citation(case, original["id"], CANONICAL, None)
    assert_citation(case, b_chunk_id, NATIVE_CANONICAL, artifact)
    a2_chunk = TextChunk.objects.filter(representation_generation=a2).first()
    assert a2_chunk.pk not in {original["id"], *(row["id"] for row in b_rows)}
    assert_citation(case, a2_chunk.pk, CANONICAL, artifact)
    assert IngestionOutput.objects.filter(pk=output.pk).values().get() == frozen_output
    assert ExtractionRevision.objects.filter(pk=output.extraction_revision_id).values().get() == frozen_revision
    after = owner_snapshot(doc)
    assert after["owners"] == before["owners"]
    changed = {"full_text", "full_text_hash", "ingestion_complete"}
    assert {k: v for k, v in after["owner"].items() if k not in changed} == {
        k: v for k, v in before["owner"].items() if k not in changed}
