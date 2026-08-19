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

    monkeypatch.setattr(chunking, "get_channel_layer", lambda: DummyChannelLayer())
    monkeypatch.setattr(
        chunking,
        "get_embeddings",
        lambda texts, input_type=None: [EMBEDDING[:] for _text in texts],
    )
    monkeypatch.setattr(
        chunking,
        "get_embedding",
        lambda _text, input_type=None: EMBEDDING[:],
    )
    monkeypatch.setattr(
        chunking,
        "doc_image_data_url",
        lambda _document: "data:image/png;base64,AAAA",
    )
    monkeypatch.setattr(
        chunking, "notify_ingest_monitor_progress", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        chunking, "notify_ingest_monitor_complete", lambda *_args, **_kwargs: None
    )

    def embed_one(chunk, callback=None):
        chunk.embedding = EMBEDDING[:]
        if callback is not None:
            callback()
        return chunk.embedding

    monkeypatch.setattr(chunking.TextChunk, "get_chunk_embedding", embed_one)
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
