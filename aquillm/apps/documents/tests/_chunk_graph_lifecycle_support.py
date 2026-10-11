from __future__ import annotations

import os
import socket
import uuid

import pytest
from django.conf import settings


def _database_is_reachable() -> bool:
    database = settings.DATABASES["default"]
    try:
        with socket.create_connection(
            (database["HOST"], int(database.get("PORT") or 5432)), timeout=0.2
        ):
            return True
    except OSError:
        return False


database_required = pytest.mark.skipif(
    not _database_is_reachable()
    and os.environ.get("KG_REQUIRE_POSTGRES_TESTS") != "1",
    reason="configured PostgreSQL database is not reachable",
)

EMBEDDING = [0.0] * 1024


class DummyChannelLayer:
    async def group_send(self, *_args, **_kwargs):
        return None


def configure_chunking_runtime(monkeypatch):
    from apps.documents.tasks import chunking
    from apps.knowledge_graph.services import builds

    # Setup activations create real rebuild intents; graph build enqueue is replaced.
    monkeypatch.setattr(builds, "enqueue_document_build", lambda *a, **k: None)

    monkeypatch.setattr(chunking, "get_channel_layer", lambda: DummyChannelLayer())
    def reject_legacy_provider(*args, **kwargs):
        raise AssertionError("strict representation publication must not use legacy provider/media fallback")
    monkeypatch.setattr(chunking, "get_embeddings", reject_legacy_provider)
    monkeypatch.setattr(chunking, "get_embedding", reject_legacy_provider)
    monkeypatch.setattr(chunking, "doc_image_data_url", reject_legacy_provider)
    monkeypatch.setattr(
        chunking, "notify_ingest_monitor_progress", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        chunking, "notify_ingest_monitor_complete", lambda *_args, **_kwargs: None
    )

    monkeypatch.setattr(chunking.TextChunk, "get_chunk_embedding", reject_legacy_provider)
    monkeypatch.setenv("APP_RAG_ENABLE_IMAGE_CHUNKS", "1")
    return chunking


def persist_document(
    document_type,
    *,
    user,
    collection,
    text: str,
    ingestion_complete: bool,
    **extra,
):
    document = document_type(
        title=f"chunk lifecycle {uuid.uuid4()}",
        full_text=text,
        full_text_hash=document_type.hash_fn(text),
        collection=collection,
        ingested_by=user,
        ingestion_complete=ingestion_complete,
        **extra,
    )
    document_type.objects.bulk_create([document])
    assert document.pkid is not None
    return document


def persist_chunk(document, *, content="old chunk", number=0, start=0):
    from apps.documents.models import TextChunk

    return TextChunk.objects.create(
        doc_id=document.id,
        content=content,
        start_position=start,
        end_position=start + len(content),
        chunk_number=number,
        modality=TextChunk.Modality.TEXT,
        metadata={"fixture": "pre-existing"},
        embedding=EMBEDDING,
    )


def run_chunk_task(chunking, document, *, expected_source_hash=None):
    return chunking.create_chunks.run(
        doc_id=str(document.id),
        expected_source_hash=(
            document.full_text_hash
            if expected_source_hash is None
            else expected_source_hash
        ),
        concrete_model_label=document._meta.label_lower,
        document_pkid=int(document.pkid),
    )


def assert_rebuild_binding(document, kwargs):
    from apps.knowledge_graph.models import GraphRebuildRequest
    request = GraphRebuildRequest.objects.get(pk=kwargs["request_id"])
    assert kwargs["eval_only"] is False
    entry = kwargs["representation_request"]
    assert request.requested_documents == [entry]
    assert entry == {
        "document_id": str(document.id), "document_pkid": document.pkid,
        "model_label": document._meta.label_lower, "collection_id": document.collection_id,
        "source_hash": document.full_text_hash,
        "representation_binding": {"version": "graph-document-representation-v1",
            "generation_key": str(document.chunks.get().representation_generation_id)},
    }
    return request


@pytest.fixture
def media_storage(settings, tmp_path):
    settings.STORAGES = {"default": {"BACKEND": "django.core.files.storage.FileSystemStorage",
                                    "OPTIONS": {"location": str(tmp_path)}}}


def persist_media_document(document_type, *, user, collection, text, pixels, monkeypatch):
    """Synthetic parser output through real I3 artifacts/output/publication envelopes."""
    from dataclasses import replace
    import io
    import zipfile
    from django.core.files.uploadedfile import SimpleUploadedFile
    from apps.collections.models import CollectionPermission
    from apps.ingestion.models import IngestionBatch, IngestionBatchItem
    from aquillm.ingestion import parsers
    from aquillm.task_ingest_uploaded import run_ingest_uploaded_file
    from apps.documents.tasks.chunking import create_chunks

    is_figure = document_type.__name__ == "DocumentFigure"
    original = parsers.extract_text_payloads
    deliveries = []
    def extract(*args, **kwargs):
        outputs = original(*args, **kwargs)
        if str(args[0]).endswith(".zip"):
            payload = replace(outputs[0], title="Synthetic media", full_text=text,
                normalized_type="document_figure" if is_figure else "image_ocr", modality="image",
                media_bytes=pixels, media_content_type="image/png",
                metadata={"figure_index": 0, "source_format": "pdf"} if is_figure else {})
            outputs = [*outputs, payload] if is_figure else [payload]
        return outputs
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("parent.txt", text.encode())
    CollectionPermission.objects.get_or_create(user=user, collection=collection, permission="EDIT")
    batch = IngestionBatch.objects.create(user=user, collection=collection)
    item = IngestionBatchItem.objects.create(batch=batch,
        source_file=SimpleUploadedFile("media.zip", archive.getvalue()),
        original_filename="media.zip", content_type="application/zip")
    with monkeypatch.context() as transport:
        transport.setattr(parsers, "extract_text_payloads", extract)
        transport.setattr(create_chunks, "delay", lambda *a, **k: deliveries.append((a, k)))
        run_ingest_uploaded_file(item.pk)
    output = item.outputs.get(output_kind="figure" if is_figure else "image", status="success")
    document = document_type.objects.get(pk=output.document_pkid)
    matched = [(args, kwargs) for args, kwargs in deliveries if args[0] == str(document.id)]
    assert len(matched) == 1
    assert set(matched[0][1]) == {"publication_id", "publication_generation", "publication_attempt"}
    return document, output, matched[0]


def assert_media_generation(document, output, pixels):
    from hashlib import sha256
    from apps.documents.models import DocumentRepresentationGeneration
    from django.core.files.storage import default_storage
    document.refresh_from_db()
    chunk = document.chunks.get()
    generation = DocumentRepresentationGeneration.objects.get(pk=chunk.representation_generation_id)
    assert generation.status == "active" and generation.origin_output_id == output.pk
    assert generation.extraction_revision_id == output.extraction_revision_id
    assert generation.manifest["origin_output_key"] == output.output_key
    entry = generation.manifest["chunks"][0]
    assert entry["modality"] == "caption_only" and entry["source_text"] == document.full_text
    assert entry["media_artifact"] == output.payload["media_artifact"]
    assert entry["media_artifact"]["sha256"] == sha256(pixels).hexdigest()
    with default_storage.open(entry["media_artifact"]["storage_key"], "rb") as blob:
        assert blob.read() == pixels
    if output.output_kind == "figure":
        assert output.parent_output_id is not None
        assert output.parent_output.extraction_revision_id == output.extraction_revision_id
        assert generation.manifest["parent_output_key"] == output.parent_output.output_key
    assert chunk.content == document.full_text and chunk.embedding is None
    assert chunk.search_vector is not None
    return chunk


def build_verified_dense(document, space, monkeypatch):
    from apps.documents.services import embedding_index_store as store
    from apps.documents.services.representation_history import reference
    from apps.documents.services.representation_store import freeze_active_representations
    from apps.documents.tasks.embedding_index import rebuild_document_embeddings, build_embedding_generation
    from apps.documents.models.embedding_index import ChunkEmbedding, EmbeddingGeneration
    from lib.embeddings.document_provider import compatible_receipt
    store.persist_embedding_space(space, using="default")
    selected = freeze_active_representations((reference(document),), using="default").selections[0]
    with monkeypatch.context() as broker:
        broker.setattr(build_embedding_generation, "delay", lambda *a, **k: None)
        key = rebuild_document_embeddings(selected, space_key=space.space_key, using="default")
    assert build_embedding_generation.run(key, using="default", timeout_ms=120000) == "activated"
    generation = EmbeddingGeneration.objects.select_related("owner").get(pk=key)
    assert generation.state == "ready"
    assert generation.owner.active_generation_id == generation.key
    dense = ChunkEmbedding.objects.get(generation_id=key)
    assert compatible_receipt(dense.vector, dense.receipt, space=space,
        role="search_document", prepared_input_sha256=dense.input_sha256) == space.space_key
    assert dense.chunk_id == document.chunks.get().pk
    return dense
