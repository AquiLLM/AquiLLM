"""Current adoption preserves canonical history without certifying old artifacts."""
import pytest
from django.db import DatabaseError, transaction

from apps.documents.models import LegacyDocumentSourceSnapshot
from apps.documents.services.representation_history import capture_legacy, locked_state
from apps.documents.services.representation_inputs import manifest_hash
from .current_adoption_support import adoption_candidate as adoption_candidate, upload as upload
from .current_adoption_support import insert_candidate

pytestmark = pytest.mark.django_db


def test_current_adoption_preserves_old_text_without_certifying_historical_artifact(adoption_candidate):
    document, chunk, values, evidence = adoption_candidate
    output = insert_candidate(values, evidence)
    with transaction.atomic():
        snapshot = capture_legacy(document, locked_state(document, using="default"), using="default")
    chunk.refresh_from_db()
    assert snapshot.canonical_text == "Existing canonical table value 10"
    assert snapshot.canonical_status == "verified"
    assert snapshot.origin_output_id is None
    assert snapshot.source_artifact_id is None
    assert chunk.legacy_context_status == "verified"
    assert (chunk.legacy_artifact_status, chunk.legacy_artifact_failure_code) == ("unavailable", "artifact_unknown")
    output.refresh_from_db()
    assert getattr(output, "provenance_kind", None) == "current_adoption"


def test_existing_snapshot_is_not_rewritten_by_later_adoption(adoption_candidate):
    document, chunk, values, evidence = adoption_candidate
    with transaction.atomic():
        state = locked_state(document, using="default")
        original = capture_legacy(document, state, using="default")
    insert_candidate(values, evidence)
    with transaction.atomic():
        after = capture_legacy(document, state, using="default")
    assert after.pk == original.pk
    assert after.origin_output_id is None
    assert LegacyDocumentSourceSnapshot.objects.count() == 1


def test_raw_snapshot_cannot_attach_current_adoption_as_historical_origin(adoption_candidate):
    document, _, values, evidence = adoption_candidate
    output = insert_candidate(values, evidence)
    with transaction.atomic():
        state = locked_state(document, using="default")
    fields = dict(canonical_text=document.full_text, canonical_sha256=document.full_text_hash,
        collection_id_at_capture=document.collection_id,
        source_artifact_id=output.source_occurrence.artifact_id,
        source_occurrence_id=output.source_occurrence_id,
        extraction_revision_id=output.extraction_revision_id, origin_output_id=output.pk,
        source_binding=output.payload["source_artifact"], media_binding=None,
        source_content_type="application/pdf", canonical_status="verified", canonical_failure_code="")
    row = LegacyDocumentSourceSnapshot(owner=state, captured_source_hash=document.full_text_hash,
        manifest_sha256=manifest_hash({"version": "legacy-canonical-snapshot-v1", **fields}), **fields)
    with pytest.raises(DatabaseError), transaction.atomic():
        LegacyDocumentSourceSnapshot.objects.bulk_create([row])


def test_native_ingestion_default_keeps_historical_artifact_provenance(upload):
    from aquillm.task_ingest_uploaded import run_ingest_uploaded_file
    from django.apps import apps
    item = upload((("native.txt", b"Original ingestion text"),))
    run_ingest_uploaded_file(item.pk)
    item.refresh_from_db()
    assert item.outputs.filter(status="success").exists(), (
        item.status, item.error_message,
        list(item.outputs.values_list("status", "error_code")),
    )
    row = item.outputs.get(status="success")
    assert getattr(row, "provenance_kind", None) == "ingestion"
    assert getattr(row, "provenance_evidence", None) == {}
    document = apps.get_model(row.document_model).objects.get(pk=row.document_pkid)
    with transaction.atomic():
        snapshot = capture_legacy(document, locked_state(document, using="default"), using="default")
    assert snapshot.origin_output_id == row.pk
