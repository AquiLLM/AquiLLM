"""File upload and ingestion-monitor API views."""
import structlog
import os

from django.contrib.auth.decorators import login_required
from django.core.exceptions import ValidationError
from django.core.validators import FileExtensionValidator
from django.db import DatabaseError
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods

from apps.collections.models import Collection
from apps.documents.models import (
    DESCENDED_FROM_DOCUMENT,
    PDFDocument,
    VTTDocument,
    document_has_raw_media,
    document_modality,
    document_provider_model,
    document_provider_name,
    DuplicateDocumentError,
)
from apps.ingestion.models import IngestionBatch
from apps.ingestion.services.upload_batches import enqueue_upload_batch_files
from aquillm.vtt import coalesce_captions, parse, to_text

logger = structlog.stdlib.get_logger(__name__)


@login_required
@require_http_methods(["POST"])
def ingest_pdf(request):
    user = request.user
    pdf_file = request.FILES.get("pdf_file")
    title = request.POST.get("title")
    collection = Collection.objects.filter(pk=request.POST.get("collection")).first()
    if not collection or not collection.user_can_edit(user):
        return JsonResponse(
            {
                "error": "Collection does not exist, was not provided, or user does not have permission to edit this collection"
            },
            status=403,
        )
    if not pdf_file:
        return JsonResponse({"error": "No PDF file provided"}, status=400)
    if not title:
        return JsonResponse({"error": "No title provided"}, status=400)
    try:
        FileExtensionValidator(["pdf"])(pdf_file)
    except ValidationError:
        return JsonResponse({"error": "Invalid file extension. Only PDF files are allowed."}, status=400)
    doc = PDFDocument(
        collection=collection,
        title=title,
        ingested_by=user,
    )
    doc.pdf_file = pdf_file
    try:
        doc.save()
    except DuplicateDocumentError as e:
        logger.error(e.message)
        return JsonResponse({"error": e.message}, status=200)
    except DatabaseError as e:
        logger.error("Database error: %s", e)
        return JsonResponse({"error": "Database error occurred while saving PDFDocument"}, status=500)

    return JsonResponse({"status_message": "Success"})


@login_required
@require_http_methods(["POST"])
def ingest_vtt(request):
    user = request.user
    vtt_file = request.FILES.get("vtt_file")
    audio_file = request.FILES.get("audio_file")
    title = request.POST.get("title")
    collection = Collection.objects.filter(pk=request.POST.get("collection")).first()
    if not collection or not collection.user_can_edit(user):
        return JsonResponse(
            {
                "error": "Collection does not exist, was not provided, or user does not have permission to edit this collection"
            },
            status=403,
        )
    if not vtt_file:
        return JsonResponse({"error": "No VTT file provided"}, status=400)
    if not title:
        return JsonResponse({"error": "No title provided"}, status=400)
    try:
        FileExtensionValidator(["vtt"])(vtt_file)
    except ValidationError:
        return JsonResponse({"error": "Invalid file extension. Only VTT files are allowed."}, status=400)
    full_text = to_text(coalesce_captions(parse(vtt_file)))

    doc = VTTDocument(
        collection=collection,
        title=title,
        ingested_by=user,
        full_text=full_text,
    )
    if audio_file:
        doc.audio_file = audio_file
    try:
        doc.save()
    except DatabaseError as e:
        logger.error("Database error: %s", e)
        return JsonResponse({"error": "Database error occurred while saving VTTDocument"}, status=500)

    return JsonResponse({"status_message": "Success"})


@login_required
@require_http_methods(["POST"])
def ingest_uploads(request):
    user = request.user
    collection = Collection.objects.filter(pk=request.POST.get("collection")).first()
    if not collection or not collection.user_can_edit(user):
        return JsonResponse(
            {
                "error": "Collection does not exist, was not provided, or user does not have permission to edit this collection"
            },
            status=403,
        )

    files = request.FILES.getlist("files")
    if not files:
        return JsonResponse({"error": "No files provided. Use multipart key 'files'."}, status=400)

    max_files = int((os.getenv("INGEST_MAX_FILES") or "50").strip())
    max_file_bytes = int((os.getenv("INGEST_MAX_FILE_BYTES") or str(50 * 1024 * 1024)).strip())
    body, status = enqueue_upload_batch_files(
        user,
        collection,
        files,
        max_files=max_files,
        max_file_bytes=max_file_bytes,
    )
    return JsonResponse(body, status=status)


@login_required
@require_http_methods(["GET"])
def ingest_uploads_status(request, batch_id):
    batch = IngestionBatch.objects.filter(id=batch_id).first()
    if batch is None:
        return JsonResponse({"error": "Batch not found."}, status=404)
    if batch.user != request.user and not batch.collection.user_can_view(request.user):
        return JsonResponse({"error": "Permission denied."}, status=403)

    items = list(
        batch.items.values(
            "id",
            "original_filename",
            "status",
            "error_message",
            "document_ids",
            "parser_metadata",
            "file_size_bytes",
            "created_at",
            "started_at",
            "finished_at",
        )
    )

    counts = {"queued": 0, "processing": 0, "success": 0, "error": 0}
    for item in items:
        state = item["status"]
        if state in counts:
            counts[state] += 1
        parser_metadata = item.get("parser_metadata") or {}
        outputs = parser_metadata.get("outputs") if isinstance(parser_metadata, dict) else []
        if not isinstance(outputs, list):
            outputs = []
        modalities = parser_metadata.get("modalities") if isinstance(parser_metadata, dict) else []
        if not isinstance(modalities, list):
            modalities = []
        if not modalities:
            modalities = sorted(
                {
                    str(output.get("modality") or "text")
                    for output in outputs
                    if isinstance(output, dict)
                }
            )
        providers = sorted(
            {
                str(output.get("provider"))
                for output in outputs
                if isinstance(output, dict) and output.get("provider")
            }
        )
        item["modalities"] = modalities
        item["providers"] = providers
        item["raw_media_saved"] = any(
            bool(output.get("raw_media_saved")) for output in outputs if isinstance(output, dict)
        )
        item["text_extracted"] = any(
            bool(output.get("text_extracted")) for output in outputs if isinstance(output, dict)
        )

    return JsonResponse(
        {
            "batch_id": batch.id,
            "collection_id": batch.collection_id,
            "counts": counts,
            "items": items,
        }
    )


@login_required
@require_http_methods(["GET"])
def ingestion_monitor(request):
    in_progress = []
    for doc_model in DESCENDED_FROM_DOCUMENT:
        in_progress.extend(
            list(
                doc_model.objects.filter(ingestion_complete=False, ingested_by=request.user).only("id", "title")
            )
        )
    protocol = "wss://" if request.is_secure() else "ws://"
    host = request.get_host()
    return JsonResponse(
        [
            {
                "documentName": doc.title,
                "documentId": str(doc.id),
                "websocketUrl": protocol + host + "/ingest/monitor/" + str(doc.id) + "/",
                "modality": document_modality(doc),
                "rawMediaSaved": document_has_raw_media(doc),
                "textExtracted": bool((doc.full_text or "").strip()),
                "provider": document_provider_name(doc),
                "providerModel": document_provider_model(doc),
            }
            for doc in in_progress
        ],
        safe=False,
    )


__all__ = [
    "ingest_pdf",
    "ingest_vtt",
    "ingest_uploads",
    "ingest_uploads_status",
    "ingestion_monitor",
]
