"""API views for document management."""
import json
import logging

from django.contrib.auth.decorators import login_required
from django.core.exceptions import ValidationError
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods

from apps.collections.models import Collection
from apps.documents.models import Document

logger = logging.getLogger(__name__)


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
        logger.error(f"Error deleting document {doc_id}: {e}")
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


__all__ = [
    'delete_document',
    'move_document',
]
