"""Real owner/source fixtures; only remote transport and Windows isolation differ.

The POSIX tests use the actual isolated parser. Windows coordinator tests replace
only validate() with its separately tested pure evaluator; all S3 configuration,
capture, immutable persistence, revisions, output guards, and PostgreSQL stay real.
"""
import importlib
import io
import os
import time
from datetime import timedelta
from types import SimpleNamespace

import pytest
from django.contrib.auth.models import User
from django.db import connection, transaction

from apps.collections.models import Collection, CollectionPermission
from apps.documents.models import (
    ChunkPublication, DocumentEmbeddingState, DocumentRepresentationGeneration,
    DocumentRepresentationState, LegacyDocumentSourceSnapshot, PDFDocument,
    TextChunk, DESCENDED_FROM_DOCUMENT,
)
from apps.documents.services.representation_contracts import DocumentRefV1
from apps.ingestion.models import IngestionBatch, IngestionBatchItem, IngestionOutput
from apps.ingestion.services import source_artifacts
from apps.knowledge_graph.models import (
    CollectionGraphMembershipState, CollectionGraphProjection, GraphArtifact,
    GraphBuildRun, GraphProjectionOutbox, GraphRebuildRequest,
)
from lib.parsers import pdf_current_validation
from .test_bounded_source_storage import Body, s3 as s3


CANONICAL = "Existing canonical value 10\n\nSecond page value 20"


def service():
    try:
        return importlib.import_module("apps.ingestion.services.current_pdf_adoption")
    except ModuleNotFoundError as exc:
        if exc.name != "apps.ingestion.services.current_pdf_adoption":
            raise
        pytest.fail("trusted same-owner current PDF adoption is not implemented")


def synthetic_pdf(*texts):
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    for text in texts:
        page = writer.add_blank_page(width=200, height=200)
        if text:
            font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica")})
            page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"):
                DictionaryObject({NameObject("/F1"): writer._add_object(font)})})
            stream = DecodedStreamObject()
            stream.set_data(b"BT /F1 12 Tf 10 100 Td <" + text.encode().hex().encode() + b"> Tj ET")
            page[NameObject("/Contents")] = writer._add_object(stream)
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


def reference(document):
    return DocumentRefV1(document._meta.label_lower, document.pkid, document.id,
                         document.collection_id, document.full_text_hash)


def owner_snapshot(document):
    """All persisted citation and publication fields, not a handpicked subset."""
    models = (DocumentRepresentationState, DocumentRepresentationGeneration,
        LegacyDocumentSourceSnapshot, DocumentEmbeddingState, ChunkPublication,
        CollectionGraphMembershipState, CollectionGraphProjection, GraphArtifact,
        GraphBuildRun, GraphProjectionOutbox, GraphRebuildRequest)
    return {
        "owner": PDFDocument.objects.filter(pk=document.pk).values().get(),
        "owners": {model._meta.label_lower: list(model.objects.order_by("pk").values_list("pk", "id"))
                   for model in DESCENDED_FROM_DOCUMENT},
        "chunks": list(TextChunk.objects.filter(doc_id=document.id).order_by("pk").values()),
        "pointers": {model._meta.label_lower: list(model.objects.order_by("pk").values())
                     for model in models},
    }


def ledger_counts():
    return tuple(model.objects.count() for model in (IngestionBatch, IngestionBatchItem, IngestionOutput))


def existing_representation(case, *, active):
    """Persist a real guarded canonical generation for admission rejection tests."""
    from apps.documents.services.representation_activation import activate_representation, stage_representation
    from apps.documents.services.representation_contracts import BuildLeaseV1
    from apps.documents.services.representation_history import capture_legacy, locked_state
    from apps.documents.services.representation_inputs import manifest_hash
    from apps.documents.services.representation_snapshot_build import snapshot_chunks, snapshot_manifest

    with transaction.atomic(), connection.cursor() as cursor:
        cursor.execute("SELECT clock_timestamp()")
        now = cursor.fetchone()[0]
        state = locked_state(case.document, using="default")
        source = capture_legacy(case.document, state, using="default")
        key, chunks = snapshot_chunks(CANONICAL)
        manifest = snapshot_manifest(case.ref, source, chunks, key)
        generation = DocumentRepresentationGeneration.objects.create(
            owner=state, collection_id=case.collection.pk, input_kind="canonical_snapshot",
            source_snapshot=source, source_revision_sha256=source.manifest_sha256,
            parser_key="canonical-snapshot-v1", chunker_key=key, preparation_policy_key="canonical-snapshot-v1",
            canonical_text=CANONICAL, canonical_sha256=case.ref.source_hash, extraction_quality="partial",
            issue_codes=["structure_unavailable"], predecessor_source_hash=case.ref.source_hash,
            predecessor_collection_id=case.ref.collection_id, manifest=manifest,
            manifest_sha256=manifest_hash(manifest), expected_count=len(chunks),
            lease_expires_at=now + timedelta(hours=1))
    if active:
        lease = BuildLeaseV1(generation.pk, generation.lease_token)
        stage_representation(lease, chunks=chunks, using="default")
        assert activate_representation(lease, expected_active_key=None, using="default") == "activated"
        generation.refresh_from_db()
    return generation


def on_first_source_read(case, callback):
    invoked = False

    class HookedBody(Body):
        def read(self, size=-1):
            nonlocal invoked
            result = super().read(size)
            if not invoked:
                invoked = True
                callback()
            return result

    case.transport.body_factory = HookedBody


def on_first_artifact_write(case, callback):
    class HookedBlobs(dict):
        invoked = False

        def __setitem__(self, key, value):
            super().__setitem__(key, value)
            if not self.invoked and "/source_artifacts/" in key:
                self.invoked = True
                callback()

    case.transport.blobs = HookedBlobs(case.transport.blobs)


@pytest.fixture
def pdf_adoption_case(s3, monkeypatch):
    storage, transport, _ = s3
    raw = synthetic_pdf("  Existing canonical value 10  ", "", "Second page value 20")
    transport.blobs["private/attached/current.pdf"] = raw
    actor = User.objects.create_user(username="current-adoption-operator")
    collection = Collection.objects.create(name="Current PDF adoption")
    grant = CollectionPermission.objects.create(user=actor, collection=collection, permission="EDIT")
    document = PDFDocument(title="Existing PDF owner", full_text=CANONICAL,
        pdf_file="attached/current.pdf", collection=collection, ingested_by=actor,
        source_url="https://example.invalid/synthetic-original")
    document.save(dont_rechunk=True)
    TextChunk.objects.bulk_create([TextChunk(doc_id=document.id, content=CANONICAL,
        start_position=0, end_position=len(CANONICAL), chunk_number=0,
        metadata={"citation_note": "retained synthetic metadata"})])
    monkeypatch.setattr(PDFDocument._meta.get_field("pdf_file"), "storage", storage)
    monkeypatch.setattr(source_artifacts, "default_storage", storage)
    native_validate = pdf_current_validation.validate
    if os.name != "posix":
        def windows_evaluator(policy, raw, canonical, stored_hash, deadline):
            assert time.monotonic() < deadline
            assert not connection.in_atomic_block, "parser work must precede row locks"
            return pdf_current_validation._evaluate(policy, raw, canonical, stored_hash)
        monkeypatch.setattr(pdf_current_validation, "validate", windows_evaluator)

    # The transport is the only source I/O double. It fails if the coordinator
    # holds row locks over reads, independently of which imports it uses.
    class OutsideTransactionBody(Body):
        def read(self, size=-1):
            assert not connection.in_atomic_block, "source I/O must precede row locks"
            return super().read(size)
    transport.body_factory = OutsideTransactionBody
    return SimpleNamespace(document=document, ref=reference(document), actor=actor,
        collection=collection, grant=grant, raw=raw, storage=storage, transport=transport,
        native_validate=native_validate)
