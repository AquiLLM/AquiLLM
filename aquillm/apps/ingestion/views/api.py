"""API views for document ingestion."""
import chardet
import gzip
import io
import json
import logging
import os
import tarfile
from xml.dom import minidom

import requests
from django.contrib.auth.decorators import login_required
from django.core.exceptions import PermissionDenied
from django.core.files.base import ContentFile
from django.core.validators import FileExtensionValidator
from django.core.exceptions import ValidationError
from django.db import DatabaseError
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods

from apps.collections.models import Collection
from apps.documents.models import (
    PDFDocument,
    TeXDocument,
    VTTDocument,
    DuplicateDocumentError,
    DESCENDED_FROM_DOCUMENT,
    document_modality,
    document_has_raw_media,
    document_provider_name,
    document_provider_model,
)
from apps.ingestion.models import IngestionBatch, IngestionBatchItem
from aquillm.vtt import parse, to_text, coalesce_captions
from aquillm.tasks import ingest_uploaded_file_task
from aquillm.crawler_tasks import crawl_and_ingest_webpage

logger = logging.getLogger(__name__)


def insert_one_from_arxiv(arxiv_id, collection, user):
    """Helper function to ingest a paper from arXiv."""

    def save_pdf_doc(content, title):
        doc = PDFDocument(
            collection=collection,
            title=title,
            ingested_by=user
        )
        doc.pdf_file.save(f'arxiv:{arxiv_id}.pdf', ContentFile(content), save=False)
        doc.save()

    status = {"message": "", "errors": []}
    tex_req = requests.get('https://arxiv.org/src/' + arxiv_id)
    pdf_req = requests.get('https://arxiv.org/pdf/' + arxiv_id)
    metadata_req = requests.get('http://export.arxiv.org/api/query?id_list=' + arxiv_id)

    if metadata_req.status_code == 404 or (tex_req.status_code == 404 and pdf_req.status_code == 404):
        status["errors"].append("ERROR: 404 from ArXiv, is the DOI correct?")
    elif (tex_req.status_code not in [200, 404] or 
          pdf_req.status_code not in [200, 404] or 
          metadata_req.status_code not in [200, 404]):
        error_str = (
            f"ERROR -- DOI {arxiv_id}: LaTeX status code {tex_req.status_code}, "
            f"PDF status code {pdf_req.status_code}, metadata status code {metadata_req.status_code}"
        )
        logger.error(error_str)
        status["errors"].append(error_str)
    else:
        xmldoc = minidom.parseString(metadata_req.content)
        title = ' '.join(
            xmldoc.getElementsByTagName('entry')[0]
                 .getElementsByTagName('title')[0]
                 .firstChild.data.split()  # type: ignore
        )
        
        if tex_req.status_code == 200:
            if tex_req.content.startswith(b'%PDF'):
                status["message"] += f"Got PDF for {arxiv_id}\n"
                save_pdf_doc(tex_req.content, title)
            else:
                status["message"] += f"Got LaTeX source for {arxiv_id}\n"
                tgz_io = io.BytesIO(tex_req.content)
                tex_str = ""
                with gzip.open(tgz_io, 'rb') as gz:
                    with tarfile.open(fileobj=gz) as tar:  # type: ignore
                        for member in tar.getmembers():
                            if member.isfile() and member.name.endswith('.tex'):
                                f = tar.extractfile(member)
                                if f:
                                    tex_bytes = f.read()
                                    encoding = chardet.detect(tex_bytes)['encoding']
                                    if not encoding:
                                        if not any(x > 127 for x in tex_bytes):
                                            encoding = 'ascii'
                                        else:
                                            raise ValueError("Could not detect encoding of LaTeX source")
                                    content = tex_bytes.decode(encoding)
                                    tex_str += content + '\n\n'
                doc = TeXDocument(
                    collection=collection,
                    title=title,
                    full_text=tex_str,
                    ingested_by=user
                )
                if pdf_req.status_code == 200:
                    status["message"] += f"Got PDF for {arxiv_id}\n"
                    doc.pdf_file.save(f'arxiv:{arxiv_id}.pdf', ContentFile(pdf_req.content), save=False)
                doc.save()
        elif pdf_req.status_code == 200:
            status["message"] += f"Got PDF for {arxiv_id}\n"
            save_pdf_doc(pdf_req.content, title)
            
    return status


@login_required
@require_http_methods(["POST"])
def ingest_arxiv(request):
    user = request.user
    arxiv_id = request.POST.get('arxiv_id')
    collection = Collection.objects.filter(pk=request.POST.get('collection')).first()
    if not collection or not collection.user_can_edit(user):
        return JsonResponse({'error': 'Collection does not exist, was not provided, or user does not have permission to edit this collection'}, status=403)
    if not arxiv_id:
        return JsonResponse({'error': 'No arXiv ID provided'}, status=400)
    try:
        status = insert_one_from_arxiv(arxiv_id, collection, user)
        if status["errors"]:
            return JsonResponse({'error': status["errors"]}, status=500)
        return JsonResponse({'message': status["message"]})
    except DuplicateDocumentError as e:
        logger.error(e.message)
        return JsonResponse({'error': e.message}, status=400)
    except DatabaseError as e:
        logger.error(f"Database error: {e}")
        return JsonResponse({'error': 'Database error occurred while saving document'}, status=500)


@login_required
@require_http_methods(["POST"])
def ingest_pdf(request):
    user = request.user
    pdf_file = request.FILES.get('pdf_file')
    title = request.POST.get('title')
    collection = Collection.objects.filter(pk=request.POST.get('collection')).first()
    if not collection or not collection.user_can_edit(user):
        return JsonResponse({'error': 'Collection does not exist, was not provided, or user does not have permission to edit this collection'}, status=403)
    if not pdf_file:
        return JsonResponse({'error': 'No PDF file provided'}, status=400)
    if not title:
        return JsonResponse({'error': 'No title provided'}, status=400)
    try:
        FileExtensionValidator(['pdf'])(pdf_file)
    except ValidationError:
        return JsonResponse({'error': 'Invalid file extension. Only PDF files are allowed.'}, status=400)
    doc = PDFDocument(
        collection=collection,
        title=title,
        ingested_by=user
    )
    doc.pdf_file = pdf_file
    try:
        doc.save()
    except DuplicateDocumentError as e:
        logger.error(e.message)
        return JsonResponse({'error': e.message}, status=200)
    except DatabaseError as e:
        logger.error(f"Database error: {e}")
        return JsonResponse({'error': 'Database error occurred while saving PDFDocument'}, status=500)

    return JsonResponse({'status_message': 'Success'})


@login_required
@require_http_methods(["POST"])
def ingest_vtt(request):
    user = request.user
    vtt_file = request.FILES.get('vtt_file')
    audio_file = request.FILES.get('audio_file')
    title = request.POST.get('title')
    collection = Collection.objects.filter(pk=request.POST.get('collection')).first()
    if not collection or not collection.user_can_edit(user):
        return JsonResponse({'error': 'Collection does not exist, was not provided, or user does not have permission to edit this collection'}, status=403)
    if not vtt_file:
        return JsonResponse({'error': 'No VTT file provided'}, status=400)
    if not title:
        return JsonResponse({'error': 'No title provided'}, status=400)
    try:
        FileExtensionValidator(['vtt'])(vtt_file)
    except ValidationError:
        return JsonResponse({'error': 'Invalid file extension. Only VTT files are allowed.'}, status=400)
    full_text = to_text(coalesce_captions(parse(vtt_file)))
    
    doc = VTTDocument(
        collection=collection,
        title=title,
        ingested_by=user,
        full_text=full_text
    )
    if audio_file:
        doc.audio_file = audio_file
    try:
        doc.save()
    except DatabaseError as e:
        logger.error(f"Database error: {e}")
        return JsonResponse({'error': 'Database error occurred while saving VTTDocument'}, status=500)

    return JsonResponse({'status_message': 'Success'})


@login_required
@require_http_methods(["POST"])
def ingest_uploads(request):
    user = request.user
    collection = Collection.objects.filter(pk=request.POST.get("collection")).first()
    if not collection or not collection.user_can_edit(user):
        return JsonResponse(
            {"error": "Collection does not exist, was not provided, or user does not have permission to edit this collection"},
            status=403,
        )

    files = request.FILES.getlist("files")
    if not files:
        return JsonResponse({"error": "No files provided. Use multipart key 'files'."}, status=400)

    max_files = int((os.getenv("INGEST_MAX_FILES") or "50").strip())
    max_file_bytes = int((os.getenv("INGEST_MAX_FILE_BYTES") or str(50 * 1024 * 1024)).strip())
    if len(files) > max_files:
        return JsonResponse({"error": f"Too many files. Maximum is {max_files} per batch."}, status=400)

    batch = IngestionBatch.objects.create(user=user, collection=collection)
    queued_items: list[dict[str, object]] = []
    rejected_items: list[dict[str, object]] = []

    for upload in files:
        size = int(getattr(upload, "size", 0) or 0)
        if size <= 0:
            rejected_items.append({"filename": upload.name, "error": "Empty file."})
            continue
        if size > max_file_bytes:
            rejected_items.append(
                {
                    "filename": upload.name,
                    "error": f"File exceeds INGEST_MAX_FILE_BYTES ({max_file_bytes}).",
                }
            )
            continue

        item = IngestionBatchItem.objects.create(
            batch=batch,
            source_file=upload,
            original_filename=upload.name,
            content_type=getattr(upload, "content_type", "") or "",
            file_size_bytes=size,
            status=IngestionBatchItem.Status.QUEUED,
        )
        ingest_uploaded_file_task.delay(item.id)
        queued_items.append({"id": item.id, "filename": item.original_filename, "status": item.status})

    if not queued_items:
        batch.delete()
        return JsonResponse(
            {
                "status": "error",
                "queued_count": 0,
                "rejected_count": len(rejected_items),
                "items": [],
                "rejected": rejected_items,
            },
            status=400,
        )

    return JsonResponse(
        {
            "batch_id": batch.id,
            "status": "queued",
            "queued_count": len(queued_items),
            "rejected_count": len(rejected_items),
            "items": queued_items,
            "rejected": rejected_items,
        },
        status=202,
    )


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
@require_http_methods(['GET'])
def ingestion_monitor(request):
    in_progress = []
    for doc_model in DESCENDED_FROM_DOCUMENT:
        in_progress.extend(
            list(
                doc_model.objects.filter(ingestion_complete=False, ingested_by=request.user).only("id", "title")
            )
        )
    protocol = 'wss://' if request.is_secure() else 'ws://'
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
        safe=False
    )


@login_required
@require_http_methods(["POST"])
def ingest_webpage(request):
    """API endpoint to initiate asynchronous webpage crawling and ingestion."""
    try:
        data = json.loads(request.body)
        url = data.get('url')
        collection_id = data.get('collection_id')
        try:
            depth = int(data.get('depth', 1))
            if depth < 0:
                depth = 0
        except (ValueError, TypeError):
            depth = 1

        if not url or not collection_id:
            logger.warning("Ingest webpage request missing url or collection_id.")
            return JsonResponse({'error': 'Missing url or collection_id'}, status=400)

        if not url.startswith(('http://', 'https://')):
            logger.warning(f"Invalid URL format received: {url}")
            return JsonResponse({'error': 'Invalid URL format. Must start with http:// or https://'}, status=400)

        try:
            collection = Collection.objects.get(pk=collection_id)
        except Collection.DoesNotExist:
            logger.warning(f"Attempted ingest to non-existent collection ID: {collection_id}")
            return JsonResponse({'error': 'Collection not found'}, status=404)

        if not collection.user_can_edit(request.user):
            logger.warning(f"Permission denied for user {request.user.id} on collection {collection_id} during webpage ingest.")
            raise PermissionDenied("You do not have permission to add documents to this collection.")

        try:
            logger.info(f"Dispatching crawl_and_ingest_webpage task for URL: {url}, Collection: {collection_id}, User: {request.user.id}, Depth: {depth}")
            crawl_and_ingest_webpage.delay(url, collection_id, request.user.id, max_depth=depth)
            return JsonResponse({'message': 'Webpage crawl initiated successfully.'}, status=202)
        except Exception as task_dispatch_error:
            logger.error(f"Failed to dispatch crawl_and_ingest_webpage task for URL {url}: {task_dispatch_error}", exc_info=True)
            return JsonResponse({'error': 'Failed to initiate webpage crawl task.'}, status=500)

    except json.JSONDecodeError:
        logger.warning("Received invalid JSON in ingest_webpage request.")
        return JsonResponse({'error': 'Invalid JSON payload'}, status=400)
    except PermissionDenied as e:
        return JsonResponse({'error': str(e)}, status=403)
    except Exception as e:
        logger.error(f"Unexpected error in ingest_webpage view setup: {e}", exc_info=True)
        return JsonResponse({'error': 'An unexpected server error occurred.'}, status=500)


__all__ = [
    'insert_one_from_arxiv',
    'ingest_arxiv',
    'ingest_pdf',
    'ingest_vtt',
    'ingest_uploads',
    'ingest_uploads_status',
    'ingestion_monitor',
    'ingest_webpage',
]
