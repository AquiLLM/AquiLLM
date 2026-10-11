"""Historical media uses the cited immutable artifact under today's permission."""
from django.core.exceptions import ObjectDoesNotExist, PermissionDenied, ValidationError
from django.core.handlers.asgi import ASGIRequest
from django.http import Http404, StreamingHttpResponse
from django.utils.cache import patch_cache_control

from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.services.representation_media import load_chunk_media_binding, verified_artifact_stream


def available_chunk_media(chunk):
    """Inspect immutable metadata only; the HTTP response verifies actual bytes."""
    media = {}
    for kind in ("pdf", "image"):
        try:
            media[kind] = load_chunk_media_binding(chunk.pk, kind=kind, using=chunk._state.db)
        except (ValueError, ValidationError, ObjectDoesNotExist):
            pass
    return media


def citation_context_and_media(chunk, document, *, include_text, threshold, padding):
    from apps.documents.services.representation_citation_context import load_citation_context
    from apps.documents.services.representation_citations import HistoricalSourceUnavailable
    try:
        source = load_citation_context(chunk, document, using=chunk._state.db,
            include_text=include_text, threshold=threshold, padding=padding)
    except HistoricalSourceUnavailable:
        return None, {}
    return source, available_chunk_media(chunk)


def chunk_media_response(request, doc_id, *, kind):
    from .pages import get_doc, _AsyncObjectBodyIterator, _ObjectBodyIterator
    stream = None
    try:
        value = request.GET.get("chunk_id", "")
        if not value.isascii() or not value.isdigit() or len(value) > 18 or int(value) <= 0:
            raise SourceScopeUnavailable("historical_media_unavailable")
        doc = get_doc(request, doc_id)
        binding = load_chunk_media_binding(int(value), kind=kind, using=doc._state.db)
        if (binding.document.model_label, binding.document.pkid, binding.document.id) != (
            doc._meta.label_lower, doc.pkid, doc.id):
            raise SourceScopeUnavailable("historical_media_unavailable")
        stream = verified_artifact_stream(binding.artifact)
        # A move/revocation during storage I/O cannot retain the previous access.
        current = get_doc(request, doc_id)
        if (current._meta.label_lower, current.pkid, current.id) != (doc._meta.label_lower, doc.pkid, doc.id):
            raise SourceScopeUnavailable("historical_media_unavailable")
        iterator = (_AsyncObjectBodyIterator if isinstance(request, ASGIRequest) else _ObjectBodyIterator)(stream)
        response = StreamingHttpResponse(iterator, content_type=binding.content_type)
        response["Content-Length"] = binding.artifact.size_bytes
        response["ETag"] = '"' + binding.artifact.sha256 + '"'
        patch_cache_control(response, private=True, no_store=True)
        return response
    except PermissionDenied:
        if stream is not None:
            stream.close()
        raise
    except (ObjectDoesNotExist, SourceScopeUnavailable, ValidationError, OSError, ValueError):
        if stream is not None:
            stream.close()
        raise Http404("Historical media is unavailable") from None
