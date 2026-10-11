"""I6: rebuild a retained parser result as fresh A2, never reactivate retired A."""
import json

import pytest
from django.test import RequestFactory

from apps.collections.models import CollectionPermission
from apps.documents.models import DocumentRepresentationGeneration
from apps.documents.services import representation_activation
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.views.api import chunk_detail
from apps.knowledge_graph.models import GraphRebuildRequest
from .representation_support import retained as retained, upload as upload, reference
from .representation_rollback_support import active, chunk_records, execute, two_generations
from .representation_rollback_graph_support import capture_graph, seed_graph, assert_graph_superseded

pytestmark = pytest.mark.django_db(transaction=True)


def test_cli_rollback_creates_fresh_generation_and_retains_authorized_history(retained):
    doc, output, legacy = retained
    a, b, original = two_generations(doc, output)
    a.refresh_from_db()
    assert a.status == "retired" and b.status == "active"
    graph, graph_before = seed_graph(doc)
    old_rows = chunk_records(a) + chunk_records(b)
    old_identity = (a.manifest, a.manifest_sha256, a.retired_at)
    before = DocumentRepresentationGeneration.objects.count()
    dry = execute(doc, original.pk, original.parser_key, dry_run=True)
    assert dry["dry_run"] is True and dry["integrity_verified"] is False
    assert dry["documents"][0]["current_generation"] == str(b.pk)
    assert active(doc).pk == b.pk and DocumentRepresentationGeneration.objects.count() == before
    result = execute(doc, original.pk, original.parser_key)
    assert result["documents"][0]["status"] == "activated"
    a2 = active(doc)
    doc.refresh_from_db()
    a.refresh_from_db()
    b.refresh_from_db()
    assert a2.pk not in (a.pk, b.pk) and a2.predecessor_generation_key == b.pk
    assert a.status == b.status == "retired" and a2.status == "active"
    assert (a.manifest, a.manifest_sha256, a.retired_at) == old_identity
    assert chunk_records(a) + chunk_records(b) == old_rows
    assert a2.origin_output_id == a.origin_output_id == output.pk
    assert a2.extraction_revision_id == a.extraction_revision_id == original.pk
    assert (a2.canonical_text, a2.canonical_sha256, a2.parser_key, a2.chunker_key,
            a2.source_revision_sha256) == (a.canonical_text, a.canonical_sha256,
            a.parser_key, a.chunker_key, a.source_revision_sha256)
    assert a2.manifest["chunks"] == a.manifest["chunks"]
    assert a2.manifest["artifact"] == a.manifest["artifact"]
    current = set(doc.chunks.values_list("pk", flat=True))
    assert current and current.isdisjoint({row["pk"] for row in old_rows} | {legacy.pk})
    assert current == {row["pk"] for row in chunk_records(a2)}
    requests = list(GraphRebuildRequest.objects.filter(scope_type="document", scope_id=str(doc.id)))
    assert any(target.get("representation_binding", {}).get("generation_key") == str(a2.pk)
        for request in requests for target in request.requested_documents)
    assert_graph_superseded(graph, graph_before)
    for generation in (a, b):
        row = chunk_records(generation)[0]
        request = RequestFactory().get("/chunk/", {"include_full_text": "1"})
        request.user = doc.ingested_by
        response = chunk_detail(request, row["pk"])
        assert response.status_code == 200
        payload = json.loads(response.content)
        assert payload["source_context_status"] == "verified"
        assert payload["document"]["full_text"] == generation.canonical_text
        assert generation.canonical_text[row["start_position"]:row["end_position"]] == row["content"]
    CollectionPermission.objects.filter(user=doc.ingested_by, collection=doc.collection).delete()
    assert chunk_detail(request, chunk_records(a)[0]["pk"]).status_code == 403
    assert chunk_detail(request, chunk_records(b)[0]["pk"]).status_code == 403


def test_same_text_competing_generation_after_lease_capture_fences_rollback(retained, monkeypatch):
    doc, output, _ = retained
    a, b, original = two_generations(doc, output)
    original_stage = representation_activation.stage_representation
    captured = []
    def compete(lease, *, chunks, using):
        captured.append(lease)
        with monkeypatch.context() as nested:
            nested.setattr(representation_activation, "stage_representation", original_stage)
            assert representation_activation.build_document_representation(reference(doc),
                extraction_revision_id=b.extraction_revision_id, chunker_key=b.chunker_key,
                using=using) == "activated"
        return original_stage(lease, chunks=chunks, using=using)
    monkeypatch.setattr(representation_activation, "stage_representation", compete)
    with pytest.raises(SourceScopeUnavailable, match="stale_representation"):
        execute(doc, original.pk, original.parser_key)
    assert len(captured) == 1
    current = active(doc)
    rejected = DocumentRepresentationGeneration.objects.get(pk=captured[0].generation_key)
    assert current.pk not in (a.pk, b.pk, rejected.pk)
    assert current.predecessor_generation_key == b.pk and current.canonical_text == b.canonical_text
    assert rejected.status == "building" and rejected.predecessor_generation_key == b.pk
    assert chunk_records(rejected) == []


@pytest.mark.parametrize("failure_stage", ["lexical", "after_graph_transition"])
def test_rollback_projection_failure_preserves_current_pointer_and_durable_history(
    retained, monkeypatch, failure_stage,
):
    from apps.documents.services import lexical_publication
    doc, output, _ = retained
    a, b, original = two_generations(doc, output)
    graph, graph_before = seed_graph(doc)
    prior = chunk_records(a) + chunk_records(b)
    request_ids = set(GraphRebuildRequest.objects.values_list("pk", flat=True))
    def fail_projection(**kwargs):
        raise RuntimeError("injected rollback projection failure")
    if failure_stage == "lexical":
        monkeypatch.setattr(lexical_publication, "publish_chunk_search_vectors", fail_projection)
    else:
        original_finish = representation_activation.finish_document_representation_transition
        def finish_then_fail(*args, **kwargs):
            original_finish(*args, **kwargs)
            assert_graph_superseded(graph, graph_before)
            assert GraphRebuildRequest.objects.exclude(pk__in=request_ids).exists()
            raise RuntimeError("injected rollback projection failure")
        monkeypatch.setattr(representation_activation, "finish_document_representation_transition", finish_then_fail)
    with pytest.raises(RuntimeError, match="injected rollback projection failure"):
        execute(doc, original.pk, original.parser_key)
    doc.refresh_from_db()
    b.refresh_from_db()
    assert active(doc).pk == b.pk and b.status == "active"
    assert doc.full_text == b.canonical_text
    assert chunk_records(a) + chunk_records(b) == prior
    assert set(GraphRebuildRequest.objects.values_list("pk", flat=True)) == request_ids
    assert capture_graph(graph) == graph_before
    staged = DocumentRepresentationGeneration.objects.get(status="staged")
    assert staged.predecessor_generation_key == b.pk and staged.extraction_revision_id == original.pk
    assert set(doc.chunks.values_list("pk", flat=True)) == {row["pk"] for row in chunk_records(b)}
