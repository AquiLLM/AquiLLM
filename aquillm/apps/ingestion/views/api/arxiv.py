"""ArXiv JSON API views."""
import logging

from django.contrib.auth.decorators import login_required
from django.db import DatabaseError
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods

from apps.collections.models import Collection
from apps.documents.models import DuplicateDocumentError
from apps.ingestion.services.arxiv_ingest import insert_one_from_arxiv

logger = logging.getLogger(__name__)


@login_required
@require_http_methods(["POST"])
def ingest_arxiv(request):
    user = request.user
    arxiv_id = request.POST.get("arxiv_id")
    collection = Collection.objects.filter(pk=request.POST.get("collection")).first()
    if not collection or not collection.user_can_edit(user):
        return JsonResponse(
            {
                "error": "Collection does not exist, was not provided, or user does not have permission to edit this collection"
            },
            status=403,
        )
    if not arxiv_id:
        return JsonResponse({"error": "No arXiv ID provided"}, status=400)
    try:
        status = insert_one_from_arxiv(arxiv_id, collection, user)
        if status["errors"]:
            return JsonResponse({"error": status["errors"]}, status=500)
        return JsonResponse({"message": status["message"]})
    except DuplicateDocumentError as e:
        logger.error(e.message)
        return JsonResponse({"error": e.message}, status=400)
    except DatabaseError as e:
        logger.error("Database error: %s", e)
        return JsonResponse({"error": "Database error occurred while saving document"}, status=500)


__all__ = ["ingest_arxiv"]
