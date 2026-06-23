"""API views for document management."""
import json
import structlog

from django.contrib.auth.decorators import login_required
from django.core.cache import cache
from django.core.exceptions import ValidationError
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods

from apps.collections.models import Collection
from apps.documents.models import Document, TextChunk
from apps.documents.services.citation_narrow import narrow_citation

logger = structlog.stdlib.get_logger(__name__)


@login_required
@require_http_methods(["DELETE"])
def delete_document(request, doc_id):
    user = request.user
    
    document = Document.get_by_id(doc_id)
    
    if not document:
        return JsonResponse({'error': 'Document not found'}, status=404)
    
    title = document.title
    if not document.collection.user_can_edit(user):
        return JsonResponse({'error': 'You do not have permission to delete this document'}, status=403)
    
    try:
        document.delete()
        return JsonResponse({
            'success': True,
            'message': f'{title} deleted successfully'
        })
    except Exception as e:
        logger.error(
            "obs.documents.delete_failed",
            doc_id=doc_id,
            error=str(e),
            error_type=type(e).__name__,
        )
        return JsonResponse({'error': f'Failed to delete document: {str(e)}'}, status=500)


@require_http_methods(["POST"])
@login_required
def move_document(request, doc_id):
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    
    new_collection_id = data.get("new_collection_id")
    if new_collection_id is None:
        return JsonResponse({"error": "new_collection_id is required"}, status=400)

    document = Document.get_by_id(doc_id)
    if not document:
        return JsonResponse({"error": "Document not found"}, status=404)

    try:
        new_collection = Collection.objects.get(id=new_collection_id)
    except Collection.DoesNotExist:
        return JsonResponse({"error": "Target collection not found"}, status=404)

    try:
        document.move_to(new_collection)
    except ValidationError as e:
        return JsonResponse({"error": str(e)}, status=400)

    return JsonResponse({
        "message": "Document moved successfully",
        "document": {
            "id": str(document.id),
            "title": document.title,
            "collection": new_collection.id,
        }
    })


FULL_TEXT_WINDOW_THRESHOLD = 500_000
FULL_TEXT_WINDOW_PADDING = 5_000


@require_http_methods(["GET"])
@login_required
def chunk_detail(request, chunk_id):
    """Return a chunk plus enough document context for the citation modal.

    The modal branches on `document.has_pdf`: when true the React side fuzzy-
    matches the chunk into the PDF text layer; when false it renders
    `document.full_text` with the chunk highlighted via offsets.
    """
    chunk = TextChunk.objects.filter(pk=chunk_id).first()
    if not chunk:
        return JsonResponse({"error": "Chunk not found"}, status=404)

    try:
        doc = chunk.document
    except ValidationError:
        return JsonResponse({"error": "Chunk's document not found"}, status=404)

    if not doc.collection.user_can_view(request.user):
        return JsonResponse(
            {"error": "You don't have access to the collection containing this chunk"},
            status=403,
        )

    has_pdf = bool(
        getattr(doc, "pdf_file", None) or getattr(doc, "rendered_pdf", None)
    )

    full_text = doc.full_text or ""
    text_offset = 0
    # Window very long docs around the chunk so the response stays bounded.
    if len(full_text) > FULL_TEXT_WINDOW_THRESHOLD:
        window_start = max(0, chunk.start_position - FULL_TEXT_WINDOW_PADDING)
        window_end = min(len(full_text), chunk.end_position + FULL_TEXT_WINDOW_PADDING)
        full_text = full_text[window_start:window_end]
        text_offset = window_start

    return JsonResponse({
        "content": chunk.content,
        "chunk_number": chunk.chunk_number,
        "start_position": chunk.start_position,
        "end_position": chunk.end_position,
        "start_time": chunk.start_time,
        "document": {
            "id": str(doc.id),
            "title": doc.title,
            "type": doc.__class__.__name__,
            "has_pdf": has_pdf,
            "source_url": getattr(doc, "source_url", None),
            "full_text": full_text,
            "text_offset": text_offset,
        },
    })


CITATION_NARROW_CACHE_TTL = 60 * 60 * 24 * 7  # 7 days


@require_http_methods(["POST"])
@login_required
def citation_narrow(request):
    """Return an LLM-narrowed quote from a chunk that supports an assistant
    message's claim. Used to tighten the in-PDF highlight from the whole
    chunk to just the relevant span.

    Body: {message_uuid: string, chunk_id: int}
    Returns: {quote: string} — empty string when the LLM declines / fails
    (caller falls back to highlighting the whole chunk).
    """
    try:
        body = json.loads(request.body or b"{}")
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    message_uuid = body.get("message_uuid")
    chunk_id = body.get("chunk_id")
    if not message_uuid or not chunk_id:
        return JsonResponse(
            {"error": "message_uuid and chunk_id required"}, status=400,
        )

    from apps.chat.models import Message

    message = Message.objects.filter(message_uuid=message_uuid).select_related("conversation").first()
    if not message:
        return JsonResponse({"error": "Message not found"}, status=404)
    if message.conversation.owner_id != request.user.id:
        return JsonResponse({"error": "Forbidden"}, status=403)

    try:
        chunk = TextChunk.objects.filter(pk=int(chunk_id)).first()
    except (TypeError, ValueError):
        return JsonResponse({"error": "Invalid chunk_id"}, status=400)
    if not chunk:
        return JsonResponse({"error": "Chunk not found"}, status=404)
    try:
        doc = chunk.document
    except ValidationError:
        return JsonResponse({"error": "Chunk's document not found"}, status=404)
    if not doc.collection.user_can_view(request.user):
        return JsonResponse({"error": "Forbidden"}, status=403)

    cache_key = f"citation_narrow:{message_uuid}:{chunk.pk}"
    cached = cache.get(cache_key)
    if cached is not None:
        return JsonResponse({"quote": cached, "cached": True})

    quote = narrow_citation(message.content, chunk.content) or ""
    cache.set(cache_key, quote, CITATION_NARROW_CACHE_TTL)
    return JsonResponse({"quote": quote, "cached": False})


__all__ = [
    'chunk_detail',
    'citation_narrow',
    'delete_document',
    'move_document',
]
