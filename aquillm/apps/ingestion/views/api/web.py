"""Web crawl ingestion API."""
import json
import structlog

from django.contrib.auth.decorators import login_required
from django.core.exceptions import PermissionDenied
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods

from apps.collections.models import Collection
from apps.ingestion.services.web_ingest import schedule_webpage_crawl

logger = structlog.stdlib.get_logger(__name__)


@login_required
@require_http_methods(["POST"])
def ingest_webpage(request):
    """API endpoint to initiate asynchronous webpage crawling and ingestion."""
    try:
        data = json.loads(request.body)
        url = data.get("url")
        collection_id = data.get("collection_id")
        try:
            depth = int(data.get("depth", 1))
            if depth < 0:
                depth = 0
        except (ValueError, TypeError):
            depth = 1

        if not url or not collection_id:
            logger.warning("Ingest webpage request missing url or collection_id.")
            return JsonResponse({"error": "Missing url or collection_id"}, status=400)

        if not url.startswith(("http://", "https://")):
            logger.warning("Invalid URL format received: %s", url)
            return JsonResponse({"error": "Invalid URL format. Must start with http:// or https://"}, status=400)

        try:
            collection = Collection.objects.get(pk=collection_id)
        except Collection.DoesNotExist:
            logger.warning("Attempted ingest to non-existent collection ID: %s", collection_id)
            return JsonResponse({"error": "Collection not found"}, status=404)

        if not collection.user_can_edit(request.user):
            logger.warning(
                "Permission denied for user %s on collection %s during webpage ingest.",
                request.user.id,
                collection_id,
            )
            raise PermissionDenied("You do not have permission to add documents to this collection.")

        try:
            logger.info(
                "Dispatching crawl_and_ingest_webpage task for URL: %s, Collection: %s, User: %s, Depth: %s",
                url,
                collection_id,
                request.user.id,
                depth,
            )
            schedule_webpage_crawl(url, collection_id, request.user.id, max_depth=depth)
            return JsonResponse({"message": "Webpage crawl initiated successfully."}, status=202)
        except Exception as task_dispatch_error:
            logger.error(
                "Failed to dispatch crawl_and_ingest_webpage task for URL %s: %s",
                url,
                task_dispatch_error,
                exc_info=True,
            )
            return JsonResponse({"error": "Failed to initiate webpage crawl task."}, status=500)

    except json.JSONDecodeError:
        logger.warning("Received invalid JSON in ingest_webpage request.")
        return JsonResponse({"error": "Invalid JSON payload"}, status=400)
    except PermissionDenied as e:
        return JsonResponse({"error": str(e)}, status=403)
    except Exception as e:
        logger.error("Unexpected error in ingest_webpage view setup: %s", e, exc_info=True)
        return JsonResponse({"error": "An unexpected server error occurred."}, status=500)


__all__ = ["ingest_webpage"]
