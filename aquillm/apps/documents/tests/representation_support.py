"""Real immutable ingestion inputs; only external broker delivery is suppressed."""
import pytest

from apps.documents.models import RawTextDocument, TextChunk
from apps.documents.services.representation_contracts import DocumentRefV1
from apps.ingestion.tests.output_support import upload as upload
from apps.ingestion.services.extraction_revisions import persist_extraction_revision
from apps.ingestion.services.source_artifacts import persist_occurrence
from aquillm.task_ingest_uploaded import run_ingest_uploaded_file
from lib.parsers.source_contracts import (
    ArchiveMemberStepV1, ExtractionV1, SourceArtifactRefV1, SourceElementV1,
)


def reference(doc):
    return DocumentRefV1(doc._meta.label_lower, doc.pkid, doc.id,
                         doc.collection_id, doc.full_text_hash)


def revision(output, text, parser="parser-v2"):
    occurrence = output.source_occurrence
    ref = SourceArtifactRefV1(**output.payload["source_artifact"])
    occurrence_ref = persist_occurrence(
        ref, archive_lineage=tuple(ArchiveMemberStepV1(**v) for v in occurrence.archive_lineage),
        original_filename=occurrence.original_filename,
        content_type=occurrence.content_type, using="default",
    )
    extraction = ExtractionV1(parser, ref, text, (
        SourceElementV1("body", "paragraph", 0, len(text), None, None, None, None),
    ), "complete", ())
    return persist_extraction_revision(occurrence_ref, extraction, using="default")


@pytest.fixture
def retained(upload):
    item = upload((("source.txt", b"old table 10"),))
    run_ingest_uploaded_file(item.pk)
    output = item.outputs.get(status="success")
    doc = RawTextDocument.objects.get(pk=output.document_pkid)
    old = TextChunk.objects.bulk_create([TextChunk(
        doc_id=doc.id, content="old table 10", start_position=0,
        end_position=12, chunk_number=0,
    )])[0]
    return doc, output, old


def activate(doc, output, text="corrected table 100", parser="parser-v2"):
    from apps.documents.services.representation_activation import build_document_representation
    from apps.documents.services.representation_inputs import configured_chunker_key
    return build_document_representation(
        reference(doc), extraction_revision_id=revision(output, text, parser),
        chunker_key=configured_chunker_key(), using="default",
    )
