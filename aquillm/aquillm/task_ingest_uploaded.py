"""Ingestion batch item processing for Celery (extracted from tasks.py)."""
from __future__ import annotations

from django.core.files.base import ContentFile
from django.db import IntegrityError
from django.utils import timezone

from apps.ingestion.services.source_artifacts import bind_payload_source, SourceArtifactIntegrityError
from lib.parsers.source_contracts import source_artifact_ref

from .task_ingest_helpers import (
    fallback_media_text,
    get_or_create_figures_subcollection,
    infer_source_title_from_figure_title,
    media_uniqueness_ref,
    normalize_source_key,
    sanitize_db_text,
)


def run_ingest_uploaded_file(item_id: int) -> None:
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
        source_docs_by_occurrence: dict[str, object] = {}
        figure_collections_by_key: dict[str, object] = {}
        for index, payload in enumerate(payloads):
            had_source_binding = payload.source_artifact is not None
            if payload.source_artifact is None:
                # Transition-only producers may omit bindings for top-level files.
                # An archived PDF can never acquire its ZIP upload as PDF bytes.
                member_path = payload.metadata.get("member_path") if isinstance(payload.metadata, dict) else None
                if payload.archive_lineage or member_path or (
                    payload.normalized_type == "pdf" and item.original_filename.lower().endswith(".zip")
                ):
                    raise SourceArtifactIntegrityError("source_artifact_required")
                payload.source_artifact = source_artifact_ref(data, member_path=None, container_sha256=None)
                payload.source_bytes = data
                payload.source_filename = item.original_filename
                payload.source_content_type = item.content_type or "application/octet-stream"
            occurrence = bind_payload_source(payload, using=item._state.db or "default")
            full_text = sanitize_db_text((payload.full_text or "")).strip()
            modality = (payload.modality or "text").strip().lower()
            payload_metadata = payload.metadata if isinstance(payload.metadata, dict) else {}
            if not full_text and modality in {"image", "audio", "video"}:
                full_text = fallback_media_text(
                    item_id=item.id,
                    payload_index=index,
                    modality=modality,
                    media_bytes=payload.media_bytes,
                )
                full_text = sanitize_db_text(full_text).strip()
            if not full_text:
                continue

            doc = None
            raw_media_saved = False
            provider = payload.provider or ""
            model_name = payload.model or ""
            source_content_type = payload.media_content_type or payload_metadata.get("content_type", "")
            safe_title = sanitize_db_text(payload.title or item.original_filename)[:200]
            is_document_figure = modality == "image" and payload.normalized_type == "document_figure"

            source_doc = None
            source_key = ""
            if is_document_figure:
                source_key = occurrence.occurrence_key
                source_doc = source_docs_by_occurrence.get(source_key)
                source_title_candidates = [
                    str(payload_metadata.get("source_document_title") or ""),
                    infer_source_title_from_figure_title(payload.title or ""),
                    infer_source_title_from_figure_title(safe_title),
                ]
                # Transition-only producers can still resolve title metadata.
                # Bound members never acquire a same-title member's parent.
                if source_doc is None and not had_source_binding:
                    for candidate in source_title_candidates:
                        candidate_key = normalize_source_key(candidate)
                        source_doc = source_docs_by_key.get(candidate_key)
                        if source_doc is not None:
                            break

            if modality == "image":
                if is_document_figure:
                    figure_collection = item.batch.collection
                    if source_key:
                        source_doc_title = str(
                            getattr(source_doc, "title", "") or payload_metadata.get("source_document_title") or ""
                        )
                        if source_key not in figure_collections_by_key:
                            figure_collections_by_key[source_key] = get_or_create_figures_subcollection(
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
                        extracted_caption=sanitize_db_text(payload_metadata.get("extracted_caption", "")),
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
                pdf_filename = payload.source_filename.rsplit("/", 1)[-1]
                doc.pdf_file.save(pdf_filename, ContentFile(payload.source_bytes), save=False)
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
                        + media_uniqueness_ref(
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
                source_docs_by_occurrence[occurrence.occurrence_key] = doc
                source_keys = {
                    normalize_source_key(payload.title or ""),
                    normalize_source_key(safe_title),
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
                    "source_artifact_sha256": occurrence.artifact.sha256,
                    "source_occurrence_id": occurrence.occurrence_id,
                    "source_occurrence_key": occurrence.occurrence_key,
                    "media_artifact_sha256": payload.media_artifact.sha256 if payload.media_artifact else None,
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
        item.error_message = sanitize_db_text(str(exc))
        item.finished_at = timezone.now()
        item.save(update_fields=["status", "error_message", "finished_at"])


__all__ = ["run_ingest_uploaded_file"]
