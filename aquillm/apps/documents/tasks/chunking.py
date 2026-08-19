"""Celery task: build embeddings and TextChunk rows for a document."""
from __future__ import annotations

import structlog
import re
import uuid
from typing import Optional

from asgiref.sync import async_to_sync
from celery.states import FAILURE
from channels.layers import get_channel_layer
from django.apps import apps as django_apps
from django.core.exceptions import ObjectDoesNotExist
from django.db import DEFAULT_DB_ALIAS, transaction

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


def _source_hash(value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("expected source hash must be a lowercase SHA-256 digest")
    return value


def _exact_document(
    doc_id: object,
    *,
    concrete_model_label: str | None,
    document_pkid: int | None,
    using: str,
):
    try:
        document_id = doc_id if type(doc_id) is uuid.UUID else uuid.UUID(str(doc_id))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ObjectDoesNotExist("Document id is not a valid UUID") from exc
    if document_id.version is None:
        raise ObjectDoesNotExist("Document id is not an RFC 4122 UUID")
    if (concrete_model_label is None) != (document_pkid is None):
        raise ValueError("concrete model label and document pkid must be supplied together")

    hinted_model = None
    if concrete_model_label is not None:
        if (
            type(concrete_model_label) is not str
            or re.fullmatch(r"[a-z0-9_]+\.[a-z0-9_]+", concrete_model_label)
            is None
        ):
            raise ValueError("concrete model label must be canonical")
        if (
            type(document_pkid) is not int
            or not 0 < document_pkid < 2**63
        ):
            raise ValueError("document pkid must be a positive signed-bigint integer")
        try:
            hinted_model = django_apps.get_model(concrete_model_label)
        except (LookupError, ValueError) as exc:
            raise ObjectDoesNotExist("Concrete document model is unavailable") from exc
        if hinted_model not in DESCENDED_FROM_DOCUMENT:
            raise ObjectDoesNotExist("Concrete document model is unsupported")

    matches: list[tuple[type, int]] = []
    for model in DESCENDED_FROM_DOCUMENT:
        pkids = tuple(
            model._base_manager.using(using)
            .filter(id=document_id)
            .order_by("pkid")
            .values_list("pkid", flat=True)[:2]
        )
        matches.extend((model, pkid) for pkid in pkids)
        if len(matches) > 1:
            break

    if concrete_model_label is not None:
        model = hinted_model
        expected = [(model, document_pkid)]
        if matches != expected:
            raise ObjectDoesNotExist("Exact document identity is absent or ambiguous")
    elif len(matches) == 1:
        model, document_pkid = matches[0]
    else:
        raise ObjectDoesNotExist("Document identity is absent or ambiguous")

    document = (
        model._base_manager.using(using)
        .filter(pkid=document_pkid, id=document_id)
        .first()
    )
    if document is None:
        raise ObjectDoesNotExist("Exact document identity no longer exists")
    return document


def _duplicate_chunks(document, *, using: str) -> list[TextChunk] | None:
    duplicate = None
    for document_model in DESCENDED_FROM_DOCUMENT:
        duplicate = (
            document_model._base_manager.using(using)
            .filter(
                full_text_hash=document.full_text_hash,
                ingestion_complete=True,
            )
            .exclude(id=document.id)
            .order_by("pkid")
            .first()
        )
        if duplicate is not None:
            break
    if duplicate is None:
        return None
    chunks = [
        TextChunk(
            content=chunk.content,
            start_position=chunk.start_position,
            end_position=chunk.end_position,
            doc_id=document.id,
            chunk_number=chunk.chunk_number,
            modality=chunk.modality,
            metadata=dict(chunk.metadata) if type(chunk.metadata) is dict else {},
            embedding=chunk.embedding,
        )
        for chunk in TextChunk.objects.using(using)
        .filter(doc_id=duplicate.id, modality=TextChunk.Modality.TEXT)
        .order_by("chunk_number", "pk")
    ]
    return chunks or None


def _prepared_chunks(document) -> tuple[list[TextChunk], Optional[TextChunk]]:
    config = django_apps.get_app_config("aquillm")
    chunk_pitch = config.chunk_size - config.chunk_overlap
    if chunk_pitch <= 0:
        raise RuntimeError("configured chunk overlap must be smaller than chunk size")
    last_character = len(document.full_text) - 1
    chunks = [
        TextChunk(
            content=document.full_text[
                chunk_pitch * index : min(
                    (chunk_pitch * index) + config.chunk_size,
                    last_character + 1,
                )
            ],
            start_position=chunk_pitch * index,
            end_position=min(
                (chunk_pitch * index) + config.chunk_size,
                last_character + 1,
            ),
            doc_id=document.id,
            chunk_number=index,
            modality=TextChunk.Modality.TEXT,
        )
        for index in range(last_character // chunk_pitch + 1)
    ]
    image_chunk: Optional[TextChunk] = None
    if _env_bool("APP_RAG_ENABLE_IMAGE_CHUNKS", True) and hasattr(
        document, "image_file"
    ):
        image_data_url = doc_image_data_url(document)
        if image_data_url:
            caption_limit = _env_int("APP_RAG_IMAGE_CAPTION_CHAR_LIMIT", 800)
            caption = (document.full_text or "").strip()[:caption_limit]
            caption = caption or f"Image document: {document.title}"
            image_start = chunks[-1].end_position if chunks else 0
            image_chunk = TextChunk(
                content=caption,
                start_position=image_start,
                end_position=image_start + max(1, len(caption)),
                doc_id=document.id,
                chunk_number=len(chunks),
                modality=TextChunk.Modality.IMAGE,
                metadata={
                    "image_name": getattr(
                        getattr(document, "image_file", None), "name", ""
                    )
                },
            )
    return chunks, image_chunk


def _embed_chunks(
    document,
    chunks: list[TextChunk],
    image_chunk: Optional[TextChunk],
    *,
    embed_text: bool = True,
) -> list[TextChunk]:
    total = len(chunks) + (1 if image_chunk is not None else 0)
    done = 0
    last_progress = -1

    def send_progress(*, force: bool = False) -> None:
        nonlocal last_progress
        progress = int((done / total) * 100) if total else 100
        if force or progress != last_progress:
            last_progress = progress
            notify_ingest_monitor_progress(document.id, progress)

    if embed_text:
        try:
            embeddings = (
                get_embeddings(
                    [chunk.content for chunk in chunks],
                    input_type="search_document",
                )
                if chunks
                else []
            )
            if len(embeddings) != len(chunks):
                raise RuntimeError(
                    f"Embedding batch mismatch: expected {len(chunks)}, got {len(embeddings)}"
                )
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding
                done += 1
                send_progress()
        except Exception as exc:
            logger.warning(
                "obs.rag.batch_embed_failed",
                document_id=str(document.id),
                error_type=type(exc).__name__,
            )
            for chunk in chunks:
                chunk.get_chunk_embedding()
                done += 1
                send_progress()
    else:
        done = len(chunks)
        send_progress()

    if image_chunk is not None:
        try:
            image_chunk.get_chunk_embedding()
        except Exception as exc:
            logger.warning(
                "obs.rag.image_embed_failed",
                document_id=str(document.id),
                error_type=type(exc).__name__,
            )
            image_chunk.embedding = get_embedding(
                image_chunk.content,
                input_type="search_document",
            )
        chunks.append(image_chunk)
        done += 1
        send_progress()
    send_progress(force=True)
    return chunks


class _StaleChunkReplacement(RuntimeError):
    pass


def _commit_chunks(
    document,
    chunks: list[TextChunk],
    *,
    expected_source_hash: str,
    using: str,
) -> str:
    from apps.knowledge_graph.graph.invalidation import (
        DocumentLifecycleRef,
        prepare_document_chunk_replacement,
        schedule_post_chunk_graph_build,
    )

    reference = DocumentLifecycleRef(
        concrete_model_label=type(document)._meta.label_lower,
        document_pkid=int(document.pkid),
        document_id=document.id,
    )
    try:
        with transaction.atomic(using=using):
            affected = prepare_document_chunk_replacement(
                reference,
                (document.collection_id,),
                expected_source_hash=expected_source_hash,
                using=using,
            )
            if affected is None:
                schedule_post_chunk_graph_build(
                    document.id,
                    expected_source_hash,
                    using=using,
                )
                return "already_committed"
            if not affected:
                raise _StaleChunkReplacement
            locked = (
                type(document)._base_manager.using(using)
                .select_for_update()
                .filter(pkid=document.pkid, id=document.id)
                .first()
            )
            if locked is None or locked.full_text_hash != expected_source_hash:
                raise _StaleChunkReplacement
            TextChunk.objects.using(using).bulk_create(chunks)
            locked.ingestion_complete = True
            locked.save(
                dont_rechunk=True,
                update_fields=["ingestion_complete"],
                using=using,
            )
            schedule_post_chunk_graph_build(
                document.id,
                expected_source_hash,
                using=using,
            )
        return "committed"
    except _StaleChunkReplacement:
        return "stale"


@app.task(serializer="json", bind=True, track_started=True)
def create_chunks(
    self,
    doc_id: str,
    expected_source_hash: str | None = None,
    concrete_model_label: str | None = None,
    document_pkid: int | None = None,
):
    database_alias = DEFAULT_DB_ALIAS
    document = _exact_document(
        doc_id,
        concrete_model_label=concrete_model_label,
        document_pkid=document_pkid,
        using=database_alias,
    )
    source_hash = _source_hash(
        document.full_text_hash
        if expected_source_hash is None
        else expected_source_hash
    )
    if document.full_text_hash != source_hash:
        return "stale"
    try:
        from apps.knowledge_graph.graph.invalidation import (
            DocumentChunkState,
            DocumentLifecycleRef,
            inspect_document_chunk_state,
            schedule_post_chunk_graph_build,
        )

        reference = DocumentLifecycleRef(
            concrete_model_label=type(document)._meta.label_lower,
            document_pkid=int(document.pkid),
            document_id=document.id,
        )
        chunk_state = inspect_document_chunk_state(
            reference,
            (document.collection_id,),
            expected_source_hash=source_hash,
            using=database_alias,
        )
        if chunk_state == DocumentChunkState.STALE:
            return "stale"
        if chunk_state == DocumentChunkState.COMMITTED:
            schedule_post_chunk_graph_build(
                document.id,
                source_hash,
                using=database_alias,
            )
            notify_ingest_monitor_complete(document.id)
            return "already_committed"
        async_to_sync(get_channel_layer().group_send)(
            f"ingestion-dashboard-{document.ingested_by.id}",
            {
                "type": "document.ingestion.start",
                "documentId": str(document.id),
                "documentName": document.title,
                "modality": document_modality(document),
                "rawMediaSaved": document_has_raw_media(document),
                "textExtracted": bool((document.full_text or "").strip()),
                "provider": document_provider_name(document),
                "providerModel": document_provider_model(document),
            },
        )
        chunks = _duplicate_chunks(document, using=database_alias)
        if chunks is None:
            text_chunks, image_chunk = _prepared_chunks(document)
            chunks = _embed_chunks(document, text_chunks, image_chunk)
        else:
            _unused_text_chunks, image_chunk = _prepared_chunks(document)
            chunks = _embed_chunks(
                document,
                chunks,
                image_chunk,
                embed_text=False,
            )
        outcome = _commit_chunks(
            document,
            chunks,
            expected_source_hash=source_hash,
            using=database_alias,
        )
        if outcome != "stale":
            notify_ingest_monitor_complete(document.id)
        return outcome
    except Exception as exc:
        logger.error(
            "obs.rag.chunking_failed",
            document_id=str(document.id),
            concrete_model=type(document)._meta.label_lower,
            document_pkid=int(document.pkid),
            expected_source_hash=source_hash,
            error_type=type(exc).__name__,
        )
        self.update_state(state=FAILURE)
        raise


__all__ = ["create_chunks"]
