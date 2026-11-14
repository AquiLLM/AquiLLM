"""Synthetic current validation records, without an adoption service or parser I/O."""
from dataclasses import asdict
from hashlib import sha256
from uuid import uuid4

import pytest
from django.utils import timezone

from apps.documents.models import PDFDocument, TextChunk
from apps.ingestion.models import IngestionBatchItem, IngestionOutput, ExtractionRevision
from apps.ingestion.services.extraction_revisions import persist_extraction_revision
from apps.ingestion.services.source_artifacts import persist_artifact, persist_occurrence
from lib.parsers.source_contracts import ExtractionV1, SourceElementV1
from .output_support import upload as upload


@pytest.fixture
def adoption_candidate(upload):
    seed = upload()
    text = "Existing canonical table value 10"
    artifact = persist_artifact(b"synthetic captured source bytes", member_path=None,
                               container_sha256=None, using="default")
    occurrence = persist_occurrence(artifact, original_filename="current.pdf",
        content_type="application/pdf", archive_lineage=(), using="default")
    extraction = ExtractionV1("current-validation:legacy-upload-PDF-current-validation-v1", artifact, text,
        (SourceElementV1("body", "paragraph", 0, len(text), None, None, None, None),),
        "partial", ("structure_unavailable",))
    revision_id = persist_extraction_revision(occurrence, extraction, using="default")
    revision = ExtractionRevision.objects.get(pk=revision_id)
    document = PDFDocument(title="Existing owner", full_text=text,
        pdf_file="existing/current.pdf", collection=seed.batch.collection,
        ingested_by=seed.batch.user)
    document.save(dont_rechunk=True)
    chunk = TextChunk.objects.bulk_create([TextChunk(doc_id=document.id, content=text,
        start_position=0, end_position=len(text), chunk_number=0)])[0]
    item = IngestionBatchItem.objects.create(batch=seed.batch, status="success",
        source_file=artifact.storage_key, original_filename="current.pdf",
        content_type="application/pdf", file_size_bytes=artifact.size_bytes)
    seed.delete()
    payload = {
        "title": document.title, "full_text": text, "normalized_type": "pdf", "modality": "text",
        "provider": "", "model": "", "issue_codes": ["structure_unavailable"], "quality": "partial",
        "source_artifact": asdict(artifact), "media_artifact": None,
        "source_content_type": "application/pdf", "source_format": "pdf", "figure_index": 0,
        "extracted_caption": "", "location_metadata": {}, "collection_id": document.collection_id,
        "collection_parent_id": document.collection.parent_id,
    }
    evidence = {
        "version": "current-adoption-evidence-v1",
        "policy": {"key": "legacy-upload-PDF-current-validation-v1", "version": "1",
                   "implementation_sha256": sha256(b"synthetic policy").hexdigest(),
                   "runtime_sha256": sha256(b"synthetic dependency versions").hexdigest()},
        "document": {"model_label": document._meta.label_lower, "pkid": document.pkid,
            "id": str(document.id), "collection_id": document.collection_id,
            "source_hash": document.full_text_hash},
        "attached_reference": {"field": "pdf_file", "name": document.pdf_file.name},
        "artifact": {"sha256": artifact.sha256, "size_bytes": artifact.size_bytes},
        "occurrence_key": occurrence.occurrence_key,
        "extraction_revision": {"id": revision_id, "manifest_sha256": revision.manifest_sha256},
        "actor_id": seed.batch.user_id, "validated_at": timezone.now().isoformat(),
    }
    values = dict(item=item, output_key=uuid4().hex + uuid4().hex,
        source_occurrence_id=occurrence.occurrence_id, extraction_revision_id=revision_id,
        output_kind="text", element_key="body", payload=payload, status="success",
        document_model=document._meta.label_lower, document_pkid=document.pkid,
        document_id=document.id, document_collection_id=document.collection_id,
        document_source_hash=document.full_text_hash)
    return document, chunk, values, evidence


def candidate(values, evidence, *, kind="current_adoption"):
    row = IngestionOutput(**values)
    # Assignment also lets the RED run exercise pre-discriminator behavior.
    row.provenance_kind = kind
    row.provenance_evidence = evidence
    return row


def insert_candidate(values, evidence, *, bulk=False, kind="current_adoption"):
    row = candidate(values, evidence, kind=kind)
    if bulk:
        IngestionOutput.objects.bulk_create([row])
    else:
        row.save()
    return row
