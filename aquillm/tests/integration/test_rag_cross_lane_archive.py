"""P1 scenario 1: archive table/formula survives second-output failure and retry."""
from copy import deepcopy
from hashlib import sha256
from uuid import uuid4

import pytest
from django.core.files.storage import default_storage

from apps.documents.models import DocumentRepresentationState, RawTextDocument, TextChunk
from apps.documents.tasks.chunking import create_chunks
from apps.ingestion.services.output_ledger import reset_output
from apps.ingestion.tests.output_support import upload as upload
from aquillm.task_ingest_uploaded import run_ingest_uploaded_file
from lib.parsers.canonical_projection import elements_from_json
from lib.parsers.tests.structural_extraction_support import workbook_bytes

pytestmark = pytest.mark.django_db(transaction=True)
SECOND = "A distinct second source must survive the retry."


def test_archive_uncached_formula_and_failed_second_output_keep_exact_first_generation(upload, monkeypatch):
    data = workbook_bytes([["Product", "Domestic", "Foreign", "Sum"],
                           ["Widget", 7, None, "=50+20"]])
    item = upload((("metrics/table.xlsx", data), ("notes/second.txt", SECOND.encode())))
    delivered = []
    monkeypatch.setattr(create_chunks, "delay", lambda *args, **kwargs: delivered.append((deepcopy(args), deepcopy(kwargs))))
    original_save = RawTextDocument.save
    def fail_second(document, *args, **kwargs):
        if document.full_text == SECOND:
            raise ValueError("synthetic private materialization details")
        return original_save(document, *args, **kwargs)
    monkeypatch.setattr(RawTextDocument, "save", fail_second)
    run_ingest_uploaded_file(item.pk)
    item.refresh_from_db()
    outputs = list(item.outputs.select_related("source_occurrence__artifact", "extraction_revision").order_by("pk"))
    assert len(outputs) == 2
    first, second = outputs
    assert first.status == "success" and second.status == "error"
    assert second.error_code == "output_failed" and second.document_id is None
    assert item.status == "partial" and item.document_ids == [str(first.document_id)]
    assert "synthetic private materialization details" not in str(item.parser_metadata)
    occurrence, revision = first.source_occurrence, first.extraction_revision
    assert occurrence.member_path == "metrics/table.xlsx" and occurrence.archive_ordinal == 0
    assert occurrence.artifact.sha256 == sha256(data).hexdigest()
    assert occurrence.artifact_id != occurrence.container_id
    with default_storage.open(occurrence.artifact.immutable_storage_key, "rb") as source:
        assert source.read() == data
    assert revision.quality == "partial" and "formula_cache_unavailable" in revision.issue_codes
    assert revision.canonical_sha256 == sha256(revision.canonical_text.encode()).hexdigest()
    cells = {cell.address: cell for element in elements_from_json(revision.ordered_elements) for cell in element.cells}
    assert cells["B2"].text == "7" and cells["C2"].value_status == "blank"
    assert cells["D2"].formula == "=50+20" and cells["D2"].value_status == "uncached"
    assert "D2 formula==50+20 value=<unknown>" in revision.canonical_text
    assert "D2 formula==50+20 value=70" not in revision.canonical_text
    document = RawTextDocument.objects.get(pk=first.document_pkid)
    assert document.full_text == revision.canonical_text
    assert len(delivered) == 1
    first_args, first_kwargs = delivered[0]
    assert first_kwargs["publication_id"]
    assert create_chunks.run(*first_args, **first_kwargs) == "activated"
    state = DocumentRepresentationState.objects.get(document_id=document.id)
    generation = state.active_generation
    assert generation.origin_output_id == first.pk and generation.extraction_revision_id == revision.pk
    assert generation.extraction_quality == "partial" and generation.issue_codes == revision.issue_codes
    before = list(TextChunk.objects.filter(representation_generation=generation).order_by("pk").values())
    assert before and all(row["embedding"] is None for row in before)
    assert all(row["search_vector"] is not None for row in before)
    for row in before:
        assert revision.canonical_text[row["start_position"]:row["end_position"]] == row["content"]
    first_identity = (first.output_key, first.document_id, first.source_occurrence_id,
                      first.extraction_revision_id, deepcopy(first.payload))
    second_identity = (second.output_key, second.source_occurrence_id, second.extraction_revision_id)
    monkeypatch.setattr(RawTextDocument, "save", original_save)
    failed_generation, failed_attempt = second.generation, second.attempt
    assert not reset_output(second.output_key, generation=uuid4(),
        extraction_revision_id=second.extraction_revision_id, using="default")
    assert not reset_output(second.output_key, generation=failed_generation,
        extraction_revision_id=first.extraction_revision_id, using="default")
    second.refresh_from_db()
    assert (second.status, second.generation, second.attempt) == ("error", failed_generation, failed_attempt)
    assert reset_output(second.output_key, generation=second.generation,
        extraction_revision_id=second.extraction_revision_id, using="default")
    run_ingest_uploaded_file(item.pk)
    first.refresh_from_db()
    second.refresh_from_db()
    item.refresh_from_db()
    assert first_identity == (first.output_key, first.document_id, first.source_occurrence_id,
                             first.extraction_revision_id, first.payload)
    assert second_identity == (second.output_key, second.source_occurrence_id, second.extraction_revision_id)
    assert second.status == "success" and second.document_id != first.document_id
    assert len(delivered) == 2 and RawTextDocument.objects.count() == 2
    assert create_chunks.run(*delivered[1][0], **delivered[1][1]) == "activated"
    assert create_chunks.run(*first_args, **first_kwargs) == "stale"
    assert list(TextChunk.objects.filter(representation_generation=generation).order_by("pk").values()) == before
    assert DocumentRepresentationState.objects.get(document_id=document.id).active_generation_id == generation.pk
    assert item.document_ids == [str(first.document_id), str(second.document_id)]
    assert item.status == "partial"  # The uncached formula remains unknown after retry.
    assert "formula_cache_unavailable" in str(item.parser_metadata)
    run_ingest_uploaded_file(item.pk)
    assert len(delivered) == 2 and RawTextDocument.objects.count() == 2
