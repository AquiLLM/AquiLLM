"""Real bounded chunking failures retain honest input without publishing output."""
import pytest
from uuid import uuid4
from django.apps import apps
from django.db import DatabaseError, connection, transaction
from django.utils import timezone

from apps.documents.models import DocumentRepresentationGeneration, TextChunk
from apps.documents.services.representation_activation import build_document_representation, stage_representation
from apps.documents.services.representation_contracts import BuildLeaseV1, SourceScopeUnavailable
from apps.documents.services.representation_inputs import configured_chunker_key
from apps.ingestion.services.extraction_revisions import persist_extraction_revision
from apps.ingestion.services.source_artifacts import persist_occurrence
from lib.parsers.source_contracts import ArchiveMemberStepV1, ExtractionV1, SourceArtifactRefV1, SourceElementV1
from .representation_support import activate, reference, retained as retained, upload as upload

pytestmark = pytest.mark.django_db(transaction=True)


def capped_revision(output):
    source = output.source_occurrence
    artifact = SourceArtifactRefV1(**output.payload["source_artifact"])
    occurrence = persist_occurrence(artifact,
        archive_lineage=tuple(ArchiveMemberStepV1(**v) for v in source.archive_lineage),
        original_filename=source.original_filename, content_type=source.content_type, using="default")
    text = "Heading of 20 chars!\n" + "x" * 1500
    extraction = ExtractionV1("capped-parser-v1", artifact, text, (
        SourceElementV1("heading", "heading", 0, 21, None, None, None, None),
        SourceElementV1("body", "paragraph", 21, len(text), None, None, None, None),
    ), "complete", ())
    return persist_extraction_revision(occurrence, extraction, using="default")


def test_real_chunker_cap_records_empty_terminal_failure_and_keeps_old_active(retained, monkeypatch):
    doc, output, old = retained
    assert activate(doc, output) == "activated"
    doc.refresh_from_db()
    active = DocumentRepresentationGeneration.objects.get(status="active")
    original_pks = list(TextChunk.objects.filter(doc_id=doc.id).values_list("pk", flat=True))
    revision_id = capped_revision(output)
    config = apps.get_app_config("aquillm")
    monkeypatch.setattr(config, "chunk_size", 20)
    monkeypatch.setattr(config, "chunk_overlap", 0)
    assert build_document_representation(reference(doc), extraction_revision_id=revision_id,
        chunker_key=configured_chunker_key(), using="default") == "failed"
    failed = DocumentRepresentationGeneration.objects.get(status="failed")
    assert failed.extraction_revision_id == revision_id
    assert failed.expected_count == 0 and failed.manifest["chunks"] == []
    assert failed.failure_code == "chunker_work_limit"
    assert failed.owner.active_generation_id == active.pk
    assert list(TextChunk.objects.filter(doc_id=doc.id).values_list("pk", flat=True)) == original_pks
    assert old.pk in original_pks
    with pytest.raises(SourceScopeUnavailable):
        stage_representation(BuildLeaseV1(failed.pk, failed.lease_token), chunks=(), using="default")
    # Exercise the zero-count guard from the otherwise legal building transition.
    failed.pk, failed.status, failed.failure_code = uuid4(), "building", ""
    failed.save(force_insert=True)
    for status in ("staged", "active", "retired"):
        with pytest.raises(DatabaseError), transaction.atomic():
            DocumentRepresentationGeneration.objects.filter(pk=failed.pk).update(
                status=status, staged_at=timezone.now(), activated_at=timezone.now(), retired_at=timezone.now())
        with pytest.raises(DatabaseError), transaction.atomic(), connection.cursor() as cursor:
            cursor.execute("UPDATE aquillm_representationgeneration SET status=%s WHERE key=%s", [status, failed.pk])
