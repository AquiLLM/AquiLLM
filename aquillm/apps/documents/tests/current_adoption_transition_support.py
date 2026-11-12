"""Synthetic same-owner transition checks; no provider or graph build work."""
import json

import pytest
from django.core.files.storage import FileSystemStorage
from django.http import Http404
from django.test import RequestFactory

from apps.documents.models import (
    DocumentEmbeddingState, EmbeddingGeneration, LegacyDocumentSourceSnapshot, TextChunk,
)
from apps.documents.services import representation_activation, representation_inputs
from apps.documents.services.embedding_index_query import freeze_dense_generations
from apps.documents.services.representation_citations import load_chunk_source
from apps.documents.services.representation_store import active_chunk_scope, freeze_active_representations
from apps.documents.views.api import chunk_detail
from apps.documents.views.pages import pdf
from apps.ingestion.models import ExtractionRevision
from apps.ingestion.services import source_artifacts
from apps.ingestion.services.extraction_revisions import persist_extraction_revision
from apps.ingestion.tests.current_pdf_adoption_support import CANONICAL, reference
from apps.knowledge_graph.models import CollectionGraphProjection, GraphArtifact, GraphRebuildRequest
from lib.embeddings.tests.test_document_contracts import profile
from lib.parsers.source_contracts import SourceArtifactRefV1
from lib.parsers.structural import extract_structure

NATIVE_CANONICAL = "  Existing canonical value 10  \nSecond page value 20"


def native_revision(case, output, tmp_path, monkeypatch):
    """Mirror captured bytes into real local storage for ordinary native reads.

    The adoption fixture forbids S3Storage.open to test its bounded capture path.
    This test covers native persistence/transition, not native S3 read bounds.
    """
    from apps.documents.services import representation_media
    artifact = SourceArtifactRefV1(**output.payload["source_artifact"])
    raw = case.transport.blobs[f"private/{artifact.storage_key}"]
    assert raw == case.raw
    target = tmp_path / artifact.storage_key
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
    storage = FileSystemStorage(location=tmp_path)
    monkeypatch.setattr(source_artifacts, "default_storage", storage)
    monkeypatch.setattr(representation_media, "default_storage", storage)
    extraction = extract_structure("source.pdf", raw, artifact=artifact)
    assert extraction.canonical_text == NATIVE_CANONICAL != CANONICAL
    assert extraction.parser_key == "native-pdf-ordered-v2"
    assert (extraction.quality, extraction.issue_codes) == ("partial", ("layout_unavailable",))
    assert [(e.kind, e.page) for e in extraction.elements] == [
        ("paragraph", 1), ("separator", 2), ("paragraph", 3)]
    source = output.source_occurrence
    assert source.archive_lineage == []
    occurrence = source_artifacts.persist_occurrence(artifact, archive_lineage=(),
        original_filename=source.original_filename, content_type=source.content_type, using="default")
    assert occurrence.occurrence_id == source.pk
    revision_id = persist_extraction_revision(occurrence, extraction, using="default")
    revision = ExtractionRevision.objects.get(pk=revision_id)
    assert revision.pk != output.extraction_revision_id
    assert revision.occurrence_id == output.source_occurrence_id
    assert revision.canonical_sha256 != output.extraction_revision.canonical_sha256
    return revision, artifact


def stage_native(document, revision):
    ref = reference(document)
    origin, revision, extraction = representation_inputs.extraction_input(ref, revision.pk, using="default")
    key = representation_inputs.configured_chunker_key()
    chunks = representation_inputs.representation_chunks(origin, extraction, key)
    manifest = representation_inputs.generation_manifest(ref, origin=origin, revision=revision,
        extraction=extraction, chunks=chunks, chunker_key=key)
    lease = representation_activation.begin_representation(ref, extraction_revision_id=revision.pk,
        chunker_key=key, manifest_sha256=representation_inputs.manifest_hash(manifest),
        expected_count=len(chunks), using="default")
    representation_activation.stage_representation(lease, chunks=chunks, using="default")
    return lease, key


def assert_original_history(original):
    row = TextChunk.objects.filter(pk=original["id"]).values().get()
    enrichment = {"legacy_source_snapshot_id", "legacy_context_status", "legacy_context_failure_code",
        "legacy_artifact_status", "legacy_artifact_failure_code"}
    assert {k: v for k, v in row.items() if k not in enrichment} == {
        k: v for k, v in original.items() if k not in enrichment}
    assert (row["legacy_context_status"], row["legacy_context_failure_code"]) == ("verified", "")
    assert (row["legacy_artifact_status"], row["legacy_artifact_failure_code"]) == (
        "unavailable", "artifact_unknown")
    snapshot = LegacyDocumentSourceSnapshot.objects.get(pk=row["legacy_source_snapshot_id"])
    assert snapshot.canonical_status == "verified" and snapshot.canonical_text == CANONICAL
    assert all(getattr(snapshot, name) is None for name in (
        "source_artifact_id", "source_occurrence_id", "extraction_revision_id", "origin_output_id",
        "source_binding", "media_binding"))
    return LegacyDocumentSourceSnapshot.objects.filter(pk=snapshot.pk).values().get()


def assert_current_generation(document, generation, expected_text):
    document.refresh_from_db()
    assert document.full_text == generation.canonical_text == expected_text
    assert document.full_text_hash == generation.canonical_sha256
    state = generation.owner
    state.refresh_from_db()
    assert state.active_generation_id == generation.pk and state.legacy_closed
    snapshot = freeze_active_representations((reference(document),), using="default")
    assert snapshot.selections[0].generation_key == generation.pk
    current = TextChunk.objects.filter(representation_generation=generation)
    assert set(active_chunk_scope(TextChunk.objects.all(), snapshot=snapshot).values_list("pk", flat=True)) == set(
        current.values_list("pk", flat=True))
    assert current.exists()
    for chunk in current:
        assert expected_text[chunk.start_position:chunk.end_position] == chunk.content
        binding = chunk.search_vector_binding
        assert binding["document"]["source_hash"] == document.full_text_hash
        assert binding["representation_key"] == str(generation.pk)
        assert binding["source_revision_sha256"] == generation.source_revision_sha256
        assert binding["canonical_sha256"] == generation.canonical_sha256
        assert chunk.search_vector is not None
    return snapshot


def assert_builds_held(document, generation, snapshot):
    # Empty membership proves no readiness is fabricated; stale real vector
    # rejection remains covered by test_representation_rollback_dense.py.
    dense = freeze_dense_generations(snapshot, space_key=profile().space_key, using="default")
    assert dense.generation_keys == ()
    assert not EmbeddingGeneration.objects.exists() and not DocumentEmbeddingState.objects.exists()
    assert not GraphArtifact.objects.filter(status__in=("active", "building")).exists()
    assert not CollectionGraphProjection.objects.filter(state="ready").exists()
    target = {"document_id": str(document.id), "document_pkid": document.pkid,
        "model_label": document._meta.label_lower, "collection_id": document.collection_id,
        "source_hash": document.full_text_hash, "representation_binding": {
            "version": "graph-document-representation-v1", "generation_key": str(generation.pk)}}
    requests = list(GraphRebuildRequest.objects.filter(scope_type="document", scope_id=str(document.id)))
    matching = [row for row in requests if row.requested_documents == [target]]
    assert len(matching) == 1
    request = matching[0]
    assert (request.status, request.document_publication_state, request.collection_publication_state) == (
        "running", "pending", "not_applicable")


def citation_request(case):
    request = RequestFactory().get("/chunk/", {"include_full_text": "1"})
    request.user = case.actor
    return request


def assert_citation(case, chunk_id, canonical, artifact):
    source = load_chunk_source(chunk_id, using="default")
    chunk = TextChunk.objects.get(pk=chunk_id)
    assert source.canonical_text == canonical and source.artifact == artifact
    assert canonical[source.start:source.end] == chunk.content
    response = chunk_detail(citation_request(case), chunk_id)
    assert response.status_code == 200
    payload = json.loads(response.content)
    assert payload["source_context_status"] == "verified"
    assert payload["document"]["full_text"] == canonical
    assert payload["document"]["has_pdf"] is (artifact is not None)
    request = RequestFactory().get("/pdf/", {"chunk_id": str(chunk_id)})
    request.user = case.actor
    if artifact is None:
        with pytest.raises(Http404):
            pdf(request, case.document.id)
    else:
        response = pdf(request, case.document.id)
        try:
            assert response.status_code == 200
            assert b"".join(response.streaming_content) == case.raw
        finally:
            response.close()
