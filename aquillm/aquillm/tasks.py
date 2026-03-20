import hashlib

from celery import shared_task
from django.core.files.base import ContentFile
from django.db import IntegrityError
from django.utils import timezone


def _media_uniqueness_ref(item_id: int, payload_index: int, modality: str, media_bytes: bytes | None) -> str:
    """Generate a stable per-payload reference for media uniqueness."""
    digest = hashlib.sha256(media_bytes or b"").hexdigest()[:12]
    return f"[ref:{item_id}:{payload_index}:{modality}:{digest}]"


def _fallback_media_text(item_id: int, payload_index: int, modality: str, media_bytes: bytes | None) -> str:
    """Generate a non-empty fallback text for media-only payloads."""
    return (
        "No text could be extracted from this media. "
        + _media_uniqueness_ref(
            item_id=item_id,
            payload_index=payload_index,
            modality=modality,
            media_bytes=media_bytes,
        )
    )


def _normalize_source_key(value: str) -> str:
    return " ".join((value or "").strip().lower().split())


def _infer_source_title_from_figure_title(title: str) -> str:
    marker = " - Figure "
    if marker not in (title or ""):
        return ""
    return title.split(marker, 1)[0].strip()


def _figures_subcollection_name(source_doc_title: str, max_len: int = 100) -> str:
    suffix = " - Figures"
    base = (source_doc_title or "").strip() or "Document"
    allowed = max_len - len(suffix)
    if allowed <= 0:
        return "Figures"[:max_len]

    if len(base) > allowed:
        digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:6]
        trunc = max(1, allowed - 7)
        base = f"{base[:trunc].rstrip()}~{digest}"
    return f"{base}{suffix}"


def _ensure_child_collection_permissions(CollectionPermission, parent_collection, child_collection) -> None:
    permissions = CollectionPermission.objects.filter(collection=parent_collection).values("user_id", "permission")
    for permission in permissions:
        CollectionPermission.objects.update_or_create(
            user_id=permission["user_id"],
            collection=child_collection,
            defaults={"permission": permission["permission"]},
        )


def _get_or_create_figures_subcollection(Collection, CollectionPermission, parent_collection, source_doc_title: str):
    child_name = _figures_subcollection_name(source_doc_title)
    try:
        child_collection, _ = Collection.objects.get_or_create(
            name=child_name,
            parent=parent_collection,
        )
    except IntegrityError:
        child_collection = Collection.objects.get(name=child_name, parent=parent_collection)

    _ensure_child_collection_permissions(
        CollectionPermission=CollectionPermission,
        parent_collection=parent_collection,
        child_collection=child_collection,
    )
    return child_collection


@shared_task(serializer="json")
def create_conversation_memories_task(conversation_id: int) -> None:
    from .models import WSConversation
    from .memory import create_episodic_memories_for_conversation

    convo = WSConversation.objects.filter(id=conversation_id).first()
    if convo is None:
        return
    create_episodic_memories_for_conversation(convo)


@shared_task(serializer="json")
def ingest_uploaded_file_task(item_id: int) -> None:
    from .ingestion.parsers import extract_text_payloads
    from .models import (
        Collection,
        CollectionPermission,
        DocumentFigure,
        ImageUploadDocument,
        IngestionBatchItem,
        MediaUploadDocument,
        PDFDocument,
        RawTextDocument,
    )

    item = IngestionBatchItem.objects.select_related("batch", "batch__collection", "batch__user").filter(id=item_id).first()
    if item is None:
        return

    item.status = IngestionBatchItem.Status.PROCESSING
    item.started_at = timezone.now()
    item.error_message = ""
    item.save(update_fields=["status", "started_at", "error_message"])

    try:
        with item.source_file.open("rb") as file_obj:
            data = file_obj.read()
        payloads = extract_text_payloads(item.original_filename, data, content_type=item.content_type or None)
        created_ids: list[str] = []
        parser_types: list[str] = []
        outputs: list[dict[str, object]] = []
        source_docs_by_key: dict[str, object] = {}
        figure_collections_by_key: dict[str, object] = {}
        for index, payload in enumerate(payloads):
            full_text = (payload.full_text or "").strip()
            modality = (payload.modality or "text").strip().lower()
            payload_metadata = payload.metadata if isinstance(payload.metadata, dict) else {}
            if not full_text and modality in {"image", "audio", "video"}:
                full_text = _fallback_media_text(
                    item_id=item.id,
                    payload_index=index,
                    modality=modality,
                    media_bytes=payload.media_bytes,
                )
            if not full_text:
                continue

            doc = None
            raw_media_saved = False
            provider = payload.provider or ""
            model_name = payload.model or ""
            source_content_type = payload.media_content_type or payload_metadata.get("content_type", "")
            safe_title = (payload.title or item.original_filename)[:200]
            is_document_figure = modality == "image" and payload.normalized_type == "document_figure"

            source_doc = None
            source_key = ""
            if is_document_figure:
                source_title_candidates = [
                    str(payload_metadata.get("source_document_title") or ""),
                    _infer_source_title_from_figure_title(payload.title or ""),
                    _infer_source_title_from_figure_title(safe_title),
                ]
                for candidate in source_title_candidates:
                    candidate_key = _normalize_source_key(candidate)
                    if not candidate_key:
                        continue
                    source_doc = source_docs_by_key.get(candidate_key)
                    if source_doc is not None:
                        source_key = candidate_key
                        break
                if not source_key:
                    for candidate in source_title_candidates:
                        candidate_key = _normalize_source_key(candidate)
                        if candidate_key:
                            source_key = candidate_key
                            break

            if modality == "image":
                if is_document_figure:
                    figure_collection = item.batch.collection
                    if source_key:
                        source_doc_title = str(
                            getattr(source_doc, "title", "") or payload_metadata.get("source_document_title") or ""
                        )
                        if source_key not in figure_collections_by_key:
                            figure_collections_by_key[source_key] = _get_or_create_figures_subcollection(
                                Collection=Collection,
                                CollectionPermission=CollectionPermission,
                                parent_collection=item.batch.collection,
                                source_doc_title=source_doc_title,
                            )
                        figure_collection = figure_collections_by_key[source_key]

                    doc = DocumentFigure(
                        title=safe_title,
                        full_text=full_text,
                        collection=figure_collection,
                        ingested_by=item.batch.user,
                        source_format=payload_metadata.get("source_format", "unknown"),
                        figure_index=payload_metadata.get("figure_index", 0),
                        extracted_caption=payload_metadata.get("extracted_caption", ""),
                        location_metadata=payload_metadata.get("location_metadata", {}),
                        source_content_type=source_content_type,
                        ocr_provider=provider,
                        ocr_model=model_name,
                    )
                    if source_doc is not None:
                        doc.parent_document = source_doc
                else:
                    doc = ImageUploadDocument(
                        title=safe_title,
                        full_text=full_text,
                        collection=item.batch.collection,
                        ingested_by=item.batch.user,
                        source_content_type=source_content_type,
                        ocr_provider=provider,
                        ocr_model=model_name,
                    )
                media_bytes = payload.media_bytes
                media_filename = payload.media_filename or item.original_filename or f"image-{item.id}-{index}.bin"
                if media_bytes:
                    doc.image_file.save(media_filename, ContentFile(media_bytes), save=False)
                    raw_media_saved = True
            elif modality in {"audio", "video"}:
                media_kind = MediaUploadDocument.MediaKind.VIDEO if modality == "video" else MediaUploadDocument.MediaKind.AUDIO
                doc = MediaUploadDocument(
                    title=safe_title,
                    full_text=full_text,
                    collection=item.batch.collection,
                    ingested_by=item.batch.user,
                    media_kind=media_kind,
                    source_content_type=source_content_type,
                    transcribe_provider=provider,
                    transcribe_model=model_name,
                )
                media_bytes = payload.media_bytes
                media_filename = payload.media_filename or item.original_filename or f"media-{item.id}-{index}.bin"
                if media_bytes:
                    doc.media_file.save(media_filename, ContentFile(media_bytes), save=False)
                    raw_media_saved = True
            elif payload.normalized_type == "pdf":
                doc = PDFDocument(
                    title=safe_title,
                    full_text=full_text,
                    collection=item.batch.collection,
                    ingested_by=item.batch.user,
                )
                pdf_filename = item.original_filename or f"document-{item.id}.pdf"
                doc.pdf_file.save(pdf_filename, ContentFile(data), save=False)
                raw_media_saved = True
            else:
                doc = RawTextDocument(
                    title=safe_title,
                    full_text=full_text,
                    collection=item.batch.collection,
                    ingested_by=item.batch.user,
                )

            try:
                doc.save()
            except IntegrityError as exc:
                # Figures can legitimately share extracted text within a source file.
                # If their text hash collides, append a per-payload ref and retry once.
                if (
                    isinstance(doc, DocumentFigure)
                    and "documentfigure_document_collection_unique" in str(exc)
                ):
                    doc.full_text = (
                        (doc.full_text.strip() + "\n")
                        + _media_uniqueness_ref(
                            item_id=item.id,
                            payload_index=index,
                            modality=modality,
                            media_bytes=payload.media_bytes,
                        )
                    )
                    doc.save()
                else:
                    raise

            if not is_document_figure:
                source_keys = {
                    _normalize_source_key(payload.title or ""),
                    _normalize_source_key(safe_title),
                }
                for candidate_key in source_keys:
                    if candidate_key:
                        source_docs_by_key[candidate_key] = doc

            created_ids.append(str(doc.id))
            parser_types.append(payload.normalized_type)
            outputs.append(
                {
                    "document_id": str(doc.id),
                    "document_model": doc.__class__.__name__,
                    "title": doc.title,
                    "normalized_type": payload.normalized_type,
                    "modality": modality,
                    "provider": provider,
                    "model": model_name,
                    "raw_media_saved": raw_media_saved,
                    "text_extracted": bool(full_text),
                }
            )

        if not created_ids:
            raise RuntimeError("No text could be extracted from this file.")

        item.document_ids = created_ids
        item.parser_metadata = {
            "normalized_types": parser_types,
            "modalities": sorted({str(output.get("modality") or "text") for output in outputs}),
            "outputs": outputs,
        }
        item.status = IngestionBatchItem.Status.SUCCESS
        item.finished_at = timezone.now()
        item.save(update_fields=["document_ids", "parser_metadata", "status", "finished_at"])
    except Exception as exc:
        item.status = IngestionBatchItem.Status.ERROR
        item.error_message = str(exc)
        item.finished_at = timezone.now()
        item.save(update_fields=["status", "error_message", "finished_at"])
