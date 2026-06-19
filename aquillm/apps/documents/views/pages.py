"""Page views for document management."""
import structlog
import mimetypes

from django.contrib.auth.decorators import login_required
from django.core.exceptions import PermissionDenied
from django.http import HttpResponse, Http404
from django.shortcuts import render
from django.views.decorators.http import require_http_methods

from apps.documents.models import DESCENDED_FROM_DOCUMENT, TextChunk

logger = structlog.stdlib.get_logger(__name__)


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
    if pdf_field:
        return HttpResponse(pdf_field, content_type='application/pdf')
    raise Http404("Requested document does not have an associated PDF")


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
