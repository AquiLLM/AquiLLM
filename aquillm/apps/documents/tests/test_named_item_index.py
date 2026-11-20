"""Exercise the write and read boundaries on real PDF rows, including legacy owners."""

from hashlib import md5
from dataclasses import replace
from io import StringIO
import json
from uuid import uuid4

import pytest
from django.contrib.auth.models import User
from django.core.management import call_command
from django.core.management.base import CommandError
from django.db import connection
from django.test.utils import CaptureQueriesContext

from apps.collections.models import Collection
from apps.documents.models import PDFDocument, PDFNamedItemIndex, TextChunk
from apps.documents.services.named_item_lookup import (
    lookup_named_item_candidates,
    validate_named_item_candidates,
)
from apps.documents.services.named_item_projection import digest
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.services.representation_history import reference
from .graph_metadata_test_support import existing_query
from apps.ingestion.tests.output_support import upload as upload

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.fixture
def pdf(settings):
    settings.RAG_NAMED_ITEM_INDEX_ENABLED = True
    settings.DOCUMENT_EMBEDDING_SPACE_KEY = None
    doc = PDFDocument(
        title="Field guide",
        full_text="Intro.\n• Moss traits: " + "green 🌿 " * 20,
        collection=Collection.objects.create(name="Fieldwork"),
        ingested_by=User.objects.create(username="field-researcher"),
    )
    doc.save(dont_rechunk=True)
    TextChunk.objects.bulk_create(
        [
            TextChunk(
                doc_id=doc.id,
                content=doc.full_text,
                start_position=0,
                end_position=len(doc.full_text),
                chunk_number=0,
            )
        ]
    )
    return doc


def backfill(**kwargs):
    out = StringIO()
    call_command("backfill_pdf_named_items", stdout=out, **kwargs)
    return json.loads(out.getvalue())


def test_backfill_is_bounded_idempotent_and_preserves_sources(pdf):
    original = list(TextChunk.objects.values_list("pk", "content"))
    after = backfill(batch_size=1)
    assert after == {"after_pk": pdf.pk, "ready": 1, "skipped": 0}
    first = PDFNamedItemIndex.objects.get().binding
    assert backfill(after_pk=pdf.pk)["ready"] == 0
    assert backfill()["ready"] == 1
    assert PDFNamedItemIndex.objects.get().binding == first
    assert list(TextChunk.objects.values_list("pk", "content")) == original
    with pytest.raises(CommandError):
        backfill(batch_size=101)


def test_lookup_requires_live_owner_and_exact_current_members(pdf):
    assert backfill()["ready"] == 1
    chunk = TextChunk.objects.get()
    with existing_query((pdf, reference(pdf))) as (runtime, scope, _):
        with CaptureQueriesContext(connection) as reads:
            (hit,) = lookup_named_item_candidates(scope, (pdf.id,), "Which moss?")
        metrics = [q["sql"] for q in reads]
        body_checks = [q for q in metrics if 'AS "body_sha"' in q]
        assert len(body_checks) == 1 and f'"id" IN ({chunk.pk})' in body_checks[0]
        assert not any('AS "body_hash"' in q for q in metrics)
        assert sum("AS members(id,n,lo,hi)" in q for q in metrics) == 1
        assert (hit.document_id, hit.generation_id, hit.chunk_id, hit.body_length) == (
            pdf.id,
            None,
            chunk.pk,
            len(pdf.full_text),
        )
        assert (hit.body_md5, hit.body_sha256) == (
            md5(pdf.full_text.encode()).hexdigest(),
            digest(pdf.full_text),
        )
        assert pdf.full_text[hit.label_start : hit.label_end] == "Moss traits"
        assert (hit.chunk_start, hit.chunk_end, hit.covered_start, hit.covered_end) == (
            0,
            len(pdf.full_text),
            pdf.full_text.index("Moss"),
            len(pdf.full_text),
        )
        assert hit.binding == PDFNamedItemIndex.objects.get().binding
        assert validate_named_item_candidates(scope, (hit,))
        assert not validate_named_item_candidates(
            scope, (replace(hit, label_end=hit.label_end - 1),)
        )
        assert runtime.budget.sources_used == 0
        with pytest.raises(SourceScopeUnavailable):
            lookup_named_item_candidates(scope, (uuid4(),), "moss")
        TextChunk.objects.filter(pk=chunk.pk).update(content=pdf.full_text.swapcase())
        assert not validate_named_item_candidates(scope, (hit,))
        assert lookup_named_item_candidates(scope, (pdf.id,), "moss") == ()
    assert backfill()["ready"] == 0 and not PDFNamedItemIndex.objects.exists()


def test_coordinate_change_or_withdrawn_projection_invalidates_capture(pdf):
    assert backfill()["ready"] == 1
    with existing_query((pdf, reference(pdf))) as (_, scope, _):
        (hit,) = lookup_named_item_candidates(scope, (pdf.id,), "moss")
        rows = TextChunk.objects.filter(pk=hit.chunk_id)
        rows.update(start_position=1)
        assert not validate_named_item_candidates(scope, (hit,))
        assert lookup_named_item_candidates(scope, (pdf.id,), "moss") == ()
        rows.update(start_position=0)
        assert validate_named_item_candidates(scope, (hit,))
        PDFNamedItemIndex.objects.all().delete()
        assert not validate_named_item_candidates(scope, (hit,))


def test_feature_off_or_unpublished_owner_has_no_nominees(pdf, settings):
    with existing_query((pdf, reference(pdf))) as (_, scope, _):
        assert lookup_named_item_candidates(scope, (pdf.id,), "moss") == ()
    settings.RAG_NAMED_ITEM_INDEX_ENABLED = False
    assert lookup_named_item_candidates(None, (pdf.id,), "moss") == ()
    with pytest.raises(CommandError):
        backfill()


def test_background_activation_builds_index_for_real_pdf_extraction(upload, settings):
    from apps.documents.services.representation_activation import (
        build_document_representation,
    )
    from apps.documents.services.representation_inputs import configured_chunker_key
    from apps.ingestion.tests.current_pdf_adoption_support import synthetic_pdf
    from aquillm.task_ingest_uploaded import run_ingest_uploaded_file

    settings.RAG_NAMED_ITEM_INDEX_ENABLED = True
    settings.DOCUMENT_EMBEDDING_SPACE_KEY = None
    data = synthetic_pdf("- Moss traits: " + "green leaves " * 14)
    item = upload((("field.pdf", data),))
    run_ingest_uploaded_file(item.pk)
    output = item.outputs.get(output_kind="text", status="success")
    pdf = PDFDocument.objects.get(pk=output.document_pkid)
    assert not PDFNamedItemIndex.objects.filter(document_id=pdf.pk).exists()
    assert (
        build_document_representation(
            reference(pdf),
            extraction_revision_id=output.extraction_revision_id,
            chunker_key=configured_chunker_key(),
            using="default",
        )
        == "activated"
    )
    projection = PDFNamedItemIndex.objects.get()
    assert projection.generation_id is not None
    pdf.refresh_from_db()
    with existing_query((pdf, reference(pdf))) as (_, scope, _):
        (hit,) = lookup_named_item_candidates(scope, (pdf.id,), "moss")
        assert hit.generation_id == projection.generation_id
        assert (
            TextChunk.objects.get(pk=hit.chunk_id).representation_generation_id
            == projection.generation_id
        )
