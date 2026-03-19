"""Backward-compatible exports and helpers for legacy aquillm.models imports.

This module intentionally does not define concrete Django model classes.
All models are sourced from domain apps under ``apps.*``.
"""

from __future__ import annotations

import base64
import logging
import mimetypes
import uuid
from os import getenv
from typing import Any, Optional

from asgiref.sync import async_to_sync
from celery.states import FAILURE
from channels.layers import get_channel_layer
from django.apps import apps
from django.core.exceptions import ObjectDoesNotExist, ValidationError
from django.core.files.storage import default_storage

from .celery import app
from .utils import get_embedding, get_embeddings

from apps.chat.models import (
    ConversationFile,
    Message,
    WSConversation,
    get_default_system_prompt as _chat_default_system_prompt,
)
from apps.collections.models import Collection, CollectionPermission, CollectionQuerySet
from apps.core.models import COLOR_SCHEME_CHOICES, FONT_FAMILY_CHOICES, UserSettings
from apps.documents.models import (
    DESCENDED_FROM_DOCUMENT,
    Document,
    DocumentFigure,
    DuplicateDocumentError,
    HandwrittenNotesDocument,
    ImageUploadDocument,
    MediaUploadDocument,
    PDFDocument,
    RawTextDocument,
    TeXDocument,
    TextChunk,
    TextChunkQuerySet,
    VTTDocument,
)
from apps.ingestion.models import IngestionBatch, IngestionBatchItem
from apps.integrations.zotero.models import ZoteroConnection
from apps.memory.models import EpisodicMemory, USER_MEMORY_CATEGORY_CHOICES, UserMemoryFact
from apps.platform_admin.models import EmailWhitelist, GeminiAPIUsage

logger = logging.getLogger(__name__)


type DocumentChild = (
    PDFDocument
    | TeXDocument
    | RawTextDocument
    | VTTDocument
    | HandwrittenNotesDocument
    | ImageUploadDocument
    | MediaUploadDocument
    | DocumentFigure
)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = (getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    try:
        value = int((getenv(name) or str(default)).strip())
    except Exception:
        value = default
    return value if value > 0 else default


def _extract_image_bytes(doc: Any) -> tuple[bytes, str] | None:
    image_file = getattr(doc, "image_file", None)
    if not image_file:
        return None
    file_name = getattr(image_file, "name", "") or ""
    try:
        if file_name and default_storage.exists(file_name):
            with default_storage.open(file_name, "rb") as f:
                return f.read(), file_name
    except Exception as exc:
        logger.warning("_extract_image_bytes storage read failed for %r: %s", file_name, exc)

    if hasattr(image_file, "read"):
        position = None
        try:
            if hasattr(image_file, "tell"):
                position = image_file.tell()
            if hasattr(image_file, "seek"):
                image_file.seek(0)
            data = image_file.read()
            if isinstance(data, bytes) and data:
                return data, file_name
        except Exception as exc:
            logger.warning("_extract_image_bytes file-object read failed: %s", exc)
            return None
        finally:
            try:
                if position is not None and hasattr(image_file, "seek"):
                    image_file.seek(position)
            except Exception:
                pass
    return None


def _resize_image_to_fit(image_bytes: bytes, max_bytes: int, file_name: str = "") -> bytes | None:
    try:
        from PIL import Image
    except ImportError:
        return None

    try:
        import io as _io

        with Image.open(_io.BytesIO(image_bytes)) as img:
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")

            quality = 85
            max_dimension = 1600
            for _ in range(5):
                width, height = img.size
                if width > max_dimension or height > max_dimension:
                    if width > height:
                        new_width = max_dimension
                        new_height = int(height * (max_dimension / width))
                    else:
                        new_height = max_dimension
                        new_width = int(width * (max_dimension / height))
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                output = _io.BytesIO()
                img.save(output, format="JPEG", quality=quality, optimize=True)
                result = output.getvalue()
                if len(result) <= max_bytes:
                    return result

                max_dimension = int(max_dimension * 0.75)
                quality = max(40, quality - 10)
    except Exception as exc:
        logger.warning("Failed resizing image %r: %s", file_name, exc)
    return None


def _to_data_url(image_bytes: bytes, file_name: str = "") -> str | None:
    if not image_bytes:
        return None
    max_bytes = _env_int("APP_RAG_IMAGE_MAX_BYTES", 350_000)
    if len(image_bytes) > max_bytes:
        resized = _resize_image_to_fit(image_bytes, max_bytes, file_name)
        if resized is None:
            return None
        image_bytes = resized
        mime_type = "image/jpeg"
    else:
        mime_type = mimetypes.guess_type(file_name or "")[0] or "image/jpeg"
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _doc_image_data_url(doc: Any) -> str | None:
    extracted = _extract_image_bytes(doc)
    if not extracted:
        return None
    image_bytes, file_name = extracted
    return _to_data_url(image_bytes, file_name)


def document_modality(doc: Any) -> str:
    if hasattr(doc, "media_kind"):
        media_kind = (getattr(doc, "media_kind", "") or "").strip().lower()
        if media_kind in {"audio", "video"}:
            return media_kind
    if hasattr(doc, "image_file"):
        return "image"
    if hasattr(doc, "audio_file"):
        return "transcript"
    return "text"


def document_has_raw_media(doc: Any) -> bool:
    return bool(getattr(doc, "image_file", None) or getattr(doc, "media_file", None))


def document_provider_name(doc: Any) -> str:
    for field_name in ("ocr_provider", "transcribe_provider"):
        value = (getattr(doc, field_name, "") or "").strip()
        if value:
            return value
    return ""


def document_provider_model(doc: Any) -> str:
    for field_name in ("ocr_model", "transcribe_model"):
        value = (getattr(doc, field_name, "") or "").strip()
        if value:
            return value
    return ""


@app.task(serializer="pickle", bind=True, track_started=True)
def create_chunks(self, doc_id: str):
    channel_layer = get_channel_layer()
    doc = Document.get_by_id(uuid.UUID(doc_id))
    if not doc:
        raise ObjectDoesNotExist(f"No document with id {doc_id}")
    try:
        async_to_sync(channel_layer.group_send)(f"ingestion-dashboard-{doc.ingested_by.id}", {
            "type": "document.ingestion.start",
            "documentId": str(doc.id),
            "documentName": doc.title,
            "modality": document_modality(doc),
            "rawMediaSaved": document_has_raw_media(doc),
            "textExtracted": bool((doc.full_text or "").strip()),
            "provider": document_provider_name(doc),
            "providerModel": document_provider_model(doc),
        })

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
            async_to_sync(channel_layer.group_send)(f"document-ingest-{doc.id}", {
                "type": "document.ingest.complete",
                "complete": True,
            })
            return

        chunk_size = apps.get_app_config("aquillm").chunk_size
        overlap = apps.get_app_config("aquillm").chunk_overlap
        chunk_pitch = chunk_size - overlap
        TextChunk.objects.filter(doc_id=doc.id).delete()
        last_character = len(doc.full_text) - 1
        chunks = [
            TextChunk(
                content=doc.full_text[chunk_pitch * i: min((chunk_pitch * i) + chunk_size, last_character + 1)],
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
            image_data_url = _doc_image_data_url(doc)
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
            async_to_sync(channel_layer.group_send)(f"document-ingest-{doc.id}", {
                "type": "document.ingest.progress",
                "progress": progress,
            })

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
        async_to_sync(channel_layer.group_send)(f"document-ingest-{doc.id}", {
            "type": "document.ingest.complete",
            "complete": True,
        })
    except Exception as exc:
        logger.error("Error creating chunks for document %s: %s", doc.id, exc)
        self.update_state(state=FAILURE)
        doc.ingestion_complete = True
        doc.full_text += f"\n\nERROR DURING PROCESSING: {exc}"
        doc.save(dont_rechunk=True)
        raise


def doc_id_validator(doc_id):
    if sum([t.objects.filter(id=doc_id).exists() for t in DESCENDED_FROM_DOCUMENT]) != 1:
        raise ValidationError("Invalid Document UUID -- either no such document or multiple")


def get_default_system_prompt() -> str:
    return _chat_default_system_prompt()


__all__ = [
    "Collection",
    "CollectionQuerySet",
    "CollectionPermission",
    "Document",
    "DocumentChild",
    "PDFDocument",
    "TeXDocument",
    "RawTextDocument",
    "VTTDocument",
    "HandwrittenNotesDocument",
    "ImageUploadDocument",
    "MediaUploadDocument",
    "DocumentFigure",
    "TextChunk",
    "TextChunkQuerySet",
    "DuplicateDocumentError",
    "DESCENDED_FROM_DOCUMENT",
    "WSConversation",
    "Message",
    "ConversationFile",
    "IngestionBatch",
    "IngestionBatchItem",
    "UserMemoryFact",
    "EpisodicMemory",
    "USER_MEMORY_CATEGORY_CHOICES",
    "UserSettings",
    "COLOR_SCHEME_CHOICES",
    "FONT_FAMILY_CHOICES",
    "EmailWhitelist",
    "GeminiAPIUsage",
    "ZoteroConnection",
    "create_chunks",
    "get_default_system_prompt",
    "doc_id_validator",
    "document_modality",
    "document_has_raw_media",
    "document_provider_name",
    "document_provider_model",
    "_doc_image_data_url",
    "get_embedding",
    "get_embeddings",
]

