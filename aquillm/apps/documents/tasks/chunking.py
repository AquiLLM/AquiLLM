"""Celery task: build embeddings and TextChunk rows for a document."""
from __future__ import annotations

import structlog
import uuid
from typing import Optional

from asgiref.sync import async_to_sync
from celery.states import FAILURE
from channels.layers import get_channel_layer
from django.apps import apps as django_apps
from django.core.exceptions import ObjectDoesNotExist

from aquillm.celery import app
from aquillm.utils import get_embedding, get_embeddings
from apps.documents.models import DESCENDED_FROM_DOCUMENT, Document, TextChunk
from apps.documents.services.chunk_progress import (
    notify_ingest_monitor_complete,
    notify_ingest_monitor_progress,
)
from apps.documents.services.document_meta import (
    document_has_raw_media,
    document_modality,
    document_provider_model,
    document_provider_name,
)
from apps.documents.services.image_payloads import _env_bool, _env_int, doc_image_data_url

logger = structlog.stdlib.get_logger(__name__)


@app.task(serializer="json", bind=True, track_started=True)
def create_chunks(self, doc_id: str):
    channel_layer = get_channel_layer()
    doc = Document.get_by_id(uuid.UUID(doc_id))
    if not doc:
        raise ObjectDoesNotExist(f"No document with id {doc_id}")
    try:
        async_to_sync(channel_layer.group_send)(
            f"ingestion-dashboard-{doc.ingested_by.id}",
            {
                "type": "document.ingestion.start",
                "documentId": str(doc.id),
                "documentName": doc.title,
                "modality": document_modality(doc),
                "rawMediaSaved": document_has_raw_media(doc),
                "textExtracted": bool((doc.full_text or "").strip()),
                "provider": document_provider_name(doc),
                "providerModel": document_provider_model(doc),
            },
        )

        existing_doc_with_same_hash = None
        for doc_type in DESCENDED_FROM_DOCUMENT:
            existing_doc_with_same_hash = doc_type.objects.filter(
                full_text_hash=doc.full_text_hash,
                ingestion_complete=True,
            ).exclude(id=doc.id).first()
            if existing_doc_with_same_hash:
                break

        if existing_doc_with_same_hash:
            existing_chunks = TextChunk.objects.filter(doc_id=existing_doc_with_same_hash.id)
            TextChunk.objects.filter(doc_id=doc.id).delete()
            new_chunks = [
                TextChunk(
                    content=chunk.content,
                    start_position=chunk.start_position,
                    end_position=chunk.end_position,
                    doc_id=doc.id,
                    chunk_number=chunk.chunk_number,
                    modality=chunk.modality,
                    metadata=chunk.metadata,
                    embedding=chunk.embedding,
                )
                for chunk in existing_chunks
            ]
            TextChunk.objects.bulk_create(new_chunks)
            doc.ingestion_complete = True
            doc.save(dont_rechunk=True)
            notify_ingest_monitor_complete(doc.id)
            return

        chunk_size = django_apps.get_app_config("aquillm").chunk_size
        overlap = django_apps.get_app_config("aquillm").chunk_overlap
        chunk_pitch = chunk_size - overlap
        TextChunk.objects.filter(doc_id=doc.id).delete()
        last_character = len(doc.full_text) - 1
        chunks = [
            TextChunk(
                content=doc.full_text[chunk_pitch * i : min((chunk_pitch * i) + chunk_size, last_character + 1)],
                start_position=chunk_pitch * i,
                end_position=min((chunk_pitch * i) + chunk_size, last_character + 1),
                doc_id=doc.id,
                chunk_number=i,
                modality=TextChunk.Modality.TEXT,
            )
            for i in range(last_character // chunk_pitch + 1)
        ]

        enable_image_chunks = _env_bool("APP_RAG_ENABLE_IMAGE_CHUNKS", True)
        image_chunk: Optional[TextChunk] = None
        if enable_image_chunks and hasattr(doc, "image_file"):
            image_data_url = doc_image_data_url(doc)
            if image_data_url:
                image_caption_limit = _env_int("APP_RAG_IMAGE_CAPTION_CHAR_LIMIT", 800)
                image_caption = (doc.full_text or "").strip()[:image_caption_limit] or f"Image document: {doc.title}"
                image_start = chunks[-1].end_position if chunks else 0
                image_chunk = TextChunk(
                    content=image_caption,
                    start_position=image_start,
                    end_position=image_start + max(1, len(image_caption)),
                    doc_id=doc.id,
                    chunk_number=len(chunks),
                    modality=TextChunk.Modality.IMAGE,
                    metadata={"image_name": getattr(getattr(doc, "image_file", None), "name", "")},
                )

        n_chunks = len(chunks) + (1 if image_chunk is not None else 0)
        done_chunks = [0]
        last_progress = [-1]

        def send_progress(force: bool = False) -> None:
            progress = int((done_chunks[0] / n_chunks) * 100) if n_chunks else 100
            if not force and progress == last_progress[0]:
                return
            last_progress[0] = progress
            notify_ingest_monitor_progress(doc.id, progress)

        chunk_texts = [chunk.content for chunk in chunks]
        try:
            embeddings = get_embeddings(chunk_texts, input_type="search_document")
            if len(embeddings) != len(chunks):
                raise RuntimeError(f"Embedding batch mismatch: expected {len(chunks)}, got {len(embeddings)}")
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding
                done_chunks[0] += 1
                send_progress()
        except Exception as exc:
            logger.warning("Batch embedding failed for document %s: %s", doc.id, exc)
            for chunk in chunks:
                chunk.get_chunk_embedding()
                done_chunks[0] += 1
                send_progress()

        if image_chunk is not None:
            try:
                image_chunk.get_chunk_embedding()
            except Exception as exc:
                logger.warning("Image embedding failed for document %s: %s", doc.id, exc)
                image_chunk.embedding = get_embedding(image_chunk.content, input_type="search_document")
            chunks.append(image_chunk)
            done_chunks[0] += 1
            send_progress()

        send_progress(force=True)
        TextChunk.objects.bulk_create(chunks)
        doc.ingestion_complete = True
        doc.save(dont_rechunk=True)
        notify_ingest_monitor_complete(doc.id)
    except Exception as exc:
        logger.error("Error creating chunks for document %s: %s", doc.id, exc)
        self.update_state(state=FAILURE)
        doc.ingestion_complete = True
        doc.full_text += f"\n\nERROR DURING PROCESSING: {exc}"
        doc.save(dont_rechunk=True)
        raise


__all__ = ["create_chunks"]
