"""Durable figure outputs retain exact caption, parent revision and immutable pixels."""
from dataclasses import replace

import pytest
from django.core.files.storage import default_storage
from django.test import Client

from apps.documents.models import DocumentFigure, DocumentRepresentationGeneration, TextChunk
from apps.documents.services.representation_activation import build_document_representation
from aquillm.task_ingest_uploaded import run_ingest_uploaded_file
from .representation_support import reference, upload as upload

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.fixture
def figure(upload, monkeypatch):
    from aquillm.ingestion import parsers
    original = parsers.extract_text_payloads
    def extract(*args, **kwargs):
        outputs = original(*args, **kwargs)
        if str(args[0]).endswith(".zip"):
            outputs.append(replace(outputs[0], title="Figure", full_text="Verified figure caption",
                normalized_type="document_figure", modality="image", media_bytes=b"immutable-image-pixels",
                media_content_type="image/png", metadata={"figure_index": 0, "source_format": "pdf"}))
        return outputs
    monkeypatch.setattr(parsers, "extract_text_payloads", extract)
    item = upload((("parent.txt", b"Complete original parent extraction"),))
    run_ingest_uploaded_file(item.pk)
    output = item.outputs.get(output_kind="figure")
    doc = DocumentFigure.objects.get(pk=output.document_pkid)
    old = TextChunk.objects.bulk_create([TextChunk(doc_id=doc.id, content=doc.full_text,
        start_position=0, end_position=len(doc.full_text), chunk_number=0, modality="image")])[0]
    return doc, output, old


def test_caption_generation_preserves_parent_revision_and_media_binding(figure):
    from apps.documents.services.representation_caption import caption_chunker_key
    doc, output, old = figure
    assert build_document_representation(reference(doc), extraction_revision_id=output.extraction_revision_id,
        chunker_key=caption_chunker_key(), using="default") == "activated"
    generation = DocumentRepresentationGeneration.objects.get(status="active")
    assert generation.canonical_text == output.payload["full_text"]
    assert generation.extraction_revision.canonical_text == "Complete original parent extraction"
    assert generation.manifest["parent_output_key"] == output.parent_output.output_key
    entry = generation.manifest["chunks"][0]
    assert entry["modality"] == "caption_only"
    assert entry["media_artifact"] == output.payload["media_artifact"]
    assert entry["source_text"] == doc.full_text
    assert TextChunk.objects.filter(pk=old.pk).exists()
    assert list(doc.chunks.values_list("representation_generation_id", flat=True)) == [generation.pk]


def test_chunk_qualified_image_uses_old_immutable_bytes_and_rejects_corruption(figure):
    from apps.documents.services.representation_caption import caption_chunker_key
    doc, output, old = figure
    assert build_document_representation(reference(doc), extraction_revision_id=output.extraction_revision_id,
        chunker_key=caption_chunker_key(), using="default") == "activated"
    client = Client()
    client.force_login(doc.ingested_by)
    url = f"/aquillm/document_image/{doc.id}/?chunk_id={old.pk}"
    doc.image_file = "different-current-file.png"
    doc.save(dont_rechunk=True, update_fields=["image_file"])
    response = client.get(url)
    assert response.status_code == 200 and response["Content-Type"] == "image/png"
    assert b"".join(response.streaming_content) == b"immutable-image-pixels"
    key = output.payload["media_artifact"]["storage_key"]
    with default_storage.open(key, "wb") as target:
        target.write(b"corrupt immutable object")
    assert client.get(url).status_code == 404
