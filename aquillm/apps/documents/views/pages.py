"""Page views for document management."""
import mimetypes
from pathlib import Path

import structlog
from botocore.exceptions import ClientError
from django.contrib.auth.decorators import login_required
from django.core.exceptions import PermissionDenied
from django.http import FileResponse, Http404, HttpResponse, StreamingHttpResponse
from django.shortcuts import render
from django.utils.cache import patch_cache_control
from django.utils.http import content_disposition_header, http_date
from django.views.decorators.http import require_http_methods
from storages.backends.s3 import S3Storage
from storages.utils import clean_name

from apps.documents.models import DESCENDED_FROM_DOCUMENT, TextChunk

logger = structlog.stdlib.get_logger(__name__)

PDF_STREAM_CHUNK_SIZE = 64 * 1024


class _ObjectBodyIterator:
    """Iterate an object-storage response body without reading it eagerly."""

    def __init__(self, body):
        self.body = body
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.closed:
            raise StopIteration
        try:
            chunk = self.body.read(PDF_STREAM_CHUNK_SIZE)
        except Exception:
            self.close()
            raise
        if chunk:
            return chunk
        self.close()
        raise StopIteration

    def close(self):
        if not self.closed:
            self.closed = True
            self.body.close()


def _private_pdf_response(response, filename):
    response["Content-Disposition"] = content_disposition_header(False, filename)
    patch_cache_control(response, private=True, max_age=300)
    return response


def _s3_pdf_response(pdf_field, storage):
    key = storage._normalize_name(clean_name(pdf_field.name))
    try:
        object_response = storage.bucket.Object(key).get()
    except ClientError as exc:
        status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if status == 404:
            raise Http404("PDF file is missing from storage") from None
        raise

    body = object_response["Body"]
    content_length = object_response.get("ContentLength")
    if content_length == 0:
        body.close()
        raise Http404("PDF file is empty")

    iterator = _ObjectBodyIterator(body)
    response = StreamingHttpResponse(iterator, content_type="application/pdf")
    if content_length is not None:
        response["Content-Length"] = content_length
    if etag := object_response.get("ETag"):
        response["ETag"] = etag
    if last_modified := object_response.get("LastModified"):
        response["Last-Modified"] = http_date(last_modified.timestamp())
    return _private_pdf_response(response, Path(pdf_field.name).name)


def _filesystem_pdf_response(pdf_field, storage):
    try:
        size = storage.size(pdf_field.name)
        if size == 0:
            raise Http404("PDF file is empty")
        pdf_field.open("rb")
    except OSError:
        raise Http404("PDF file is missing from storage") from None

    response = FileResponse(
        pdf_field,
        as_attachment=False,
        filename=Path(pdf_field.name).name,
        content_type="application/pdf",
    )
    return _private_pdf_response(response, Path(pdf_field.name).name)


def get_doc(request, doc_id):
    """Helper function to get a document by ID and verify access permissions."""
    doc = None
    for t in DESCENDED_FROM_DOCUMENT:
        doc = t.objects.filter(id=doc_id).first()
        if doc:
            break
    if not doc:
        raise Http404("Requested document does not exist")
    if not doc.collection.user_can_view(request.user):
        raise PermissionDenied("You don't have access to the collection containing this document")
    return doc


@require_http_methods(['GET'])
@login_required
def pdf(request, doc_id):
    """Serve the PDF file for a document.

    PDFDocument uses `pdf_file`; TeXDocument uses `pdf_file` when compiled;
    RawTextDocument uses `rendered_pdf` populated by the web crawler.
    """
    doc = get_doc(request, doc_id)
    pdf_field = getattr(doc, 'pdf_file', None) or getattr(doc, 'rendered_pdf', None)
    if not pdf_field:
        raise Http404("Requested document does not have an associated PDF")

    storage = pdf_field.storage
    if isinstance(storage, S3Storage):
        return _s3_pdf_response(pdf_field, storage)
    return _filesystem_pdf_response(pdf_field, storage)


@require_http_methods(['GET'])
@login_required
def document_image(request, doc_id):
    """Serve the image file for an ImageUploadDocument or HandwrittenNotesDocument."""
    doc = get_doc(request, doc_id)

    image_file = getattr(doc, 'image_file', None)
    if not image_file:
        raise Http404("Requested document does not have an associated image file")

    content_type, _ = mimetypes.guess_type(image_file.name)
    if not content_type:
        content_type = 'image/jpeg'
    try:
        with image_file.open("rb") as f:
            data = f.read()
    except FileNotFoundError:
        raise Http404("Image file is missing from storage") from None
    if not data:
        raise Http404("Image file is empty")
    return HttpResponse(data, content_type=content_type)


@require_http_methods(['GET'])
@login_required
def document(request, doc_id):
    """Display a document detail page."""
    doc = get_doc(request, doc_id)
    highlight_chunk = None
    raw_chunk = request.GET.get('chunk')
    if raw_chunk is not None and raw_chunk.isdigit():
        highlight_chunk = TextChunk.objects.filter(pk=int(raw_chunk, 10), doc_id=doc_id).first()
    context = {'document': doc, 'highlight_chunk': highlight_chunk}
    return render(request, 'aquillm/document.html', context)


__all__ = [
    'get_doc',
    'pdf',
    'document_image',
    'document',
]
