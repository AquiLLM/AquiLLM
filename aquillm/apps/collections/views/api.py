"""API views for collection management."""
import json
import logging

from django.contrib.auth import get_user_model
from django.contrib.auth.decorators import login_required
from django.core.exceptions import ValidationError
from django.db import transaction
from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from django.views.decorators.http import require_http_methods

from apps.collections.models import Collection, CollectionPermission
from apps.documents.models import DESCENDED_FROM_DOCUMENT, DocumentFigure
from apps.ingestion.models import IngestionBatchItem

logger = logging.getLogger(__name__)


def _normalized_type_label(normalized_type: str) -> str:
    raw = (normalized_type or "").strip().lower()
    if not raw:
        return ""
    if raw == "document_figure":
        return "DocumentFigure"
    # Most parser normalized types are file-ish identifiers (pptx, docx, jsonl, ...).
    return raw.upper()


def _raw_text_type_overrides(collection: Collection) -> dict[str, str]:
    """Map document UUID -> parser-derived display type for RawTextDocument rows."""
    overrides: dict[str, str] = {}
    items = IngestionBatchItem.objects.filter(
        batch__collection=collection,
        status=IngestionBatchItem.Status.SUCCESS,
    ).only("parser_metadata")

    for item in items:
        parser_metadata = item.parser_metadata or {}
        if not isinstance(parser_metadata, dict):
            continue
        outputs = parser_metadata.get("outputs") or []
        if not isinstance(outputs, list):
            continue

        for output in outputs:
            if not isinstance(output, dict):
                continue
            document_id = str(output.get("document_id") or "").strip()
            normalized_type = str(output.get("normalized_type") or "").strip()
            if not document_id or not normalized_type:
                continue
            overrides[document_id] = _normalized_type_label(normalized_type)

    return overrides


def _child_collection_parent_document_ids(
    child_collection_ids: list[int],
    valid_document_ids: set[str],
    document_title_to_id: dict[str, str],
) -> dict[int, str]:
    """Map child collection id -> parent document id (when figures in that child link back to a document)."""
    if not child_collection_ids:
        return {}

    def _norm(text: str) -> str:
        return " ".join((text or "").strip().lower().split())

    def _source_title_from_figure_title(title: str) -> str:
        marker = " - Figure "
        if marker not in (title or ""):
            return ""
        return title.split(marker, 1)[0].strip()

    mapping: dict[int, str] = {}
    rows = (
        DocumentFigure.objects.filter(
            collection_id__in=child_collection_ids,
        )
        .values("collection_id", "parent_object_id", "title")
    )
    for row in rows:
        collection_id = int(row["collection_id"])
        if collection_id in mapping:
            continue
        parent_document_id = str(row["parent_object_id"] or "").strip()
        if parent_document_id and (not valid_document_ids or parent_document_id in valid_document_ids):
            mapping[collection_id] = parent_document_id
            continue

        source_title = _source_title_from_figure_title(str(row.get("title") or ""))
        if not source_title:
            continue
        inferred_id = document_title_to_id.get(_norm(source_title))
        if inferred_id:
            mapping[collection_id] = inferred_id

    return mapping


@login_required
@require_http_methods(["DELETE"])
def delete_collection(request, collection_id):
    user = request.user
    collection = get_object_or_404(Collection, id=collection_id)
    
    if not collection.user_can_manage(user):
        return JsonResponse({'error': 'You do not have permission to delete this collection'}, status=403)
    
    children_count = collection.children.count()
    documents_count = sum(len(x.objects.filter(collection=collection)) for x in DESCENDED_FROM_DOCUMENT)
    
    try:
        collection.delete()
        return JsonResponse({
            'success': True,
            'message': f'Collection deleted successfully along with {children_count} subcollections and {documents_count} documents'
        })
    except Exception as e:
        logger.error(f"Error deleting collection {collection_id}: {e}")
        return JsonResponse({'error': f'Failed to delete collection: {str(e)}'}, status=500)


@require_http_methods(['GET', 'POST'])
@login_required
def collections(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        name = data.get('name')
        if not name:
            return JsonResponse({'error': 'Name is required'}, status=400)

        with transaction.atomic():
            collection = Collection.objects.create(name=name)
            CollectionPermission.objects.create(
                collection=collection,
                user=request.user,
                permission='MANAGE'
            )

            for viewer in data.get('viewers', []):
                CollectionPermission.objects.create(
                    collection=collection,
                    user=get_user_model().objects.get(id=viewer),
                    permission='VIEW'
                )
            for editor in data.get('editors', []):
                CollectionPermission.objects.create(
                    collection=collection,
                    user=get_user_model().objects.get(id=editor),
                    permission='EDIT'
                )
            for admin in data.get('admins', []):
                CollectionPermission.objects.create(
                    collection=collection,
                    user=get_user_model().objects.get(id=admin),
                    permission='MANAGE'
                )

            return JsonResponse({
                'id': collection.id,
                'name': collection.name,
                'parent': collection.parent.id if collection.parent else None,
                'path': collection.get_path(),
                'document_count': collection.document_count(),
                'children_count': collection.children.count(),
                'permission': 'MANAGE'
            })

    # For GET requests
    colperms = CollectionPermission.objects.filter(user=request.user)
    collections_list = []
    for colperm in colperms:
        collections_list.append({
            'id': colperm.collection.id,
            'name': colperm.collection.name,
            'parent': colperm.collection.parent.id if colperm.collection.parent else None,
            'path': colperm.collection.get_path(),
            'document_count': colperm.collection.document_count(),
            'children_count': colperm.collection.children.count(),
            'permission': colperm.permission
        })
    return JsonResponse({"collections": collections_list})


@require_http_methods(["POST"])
@login_required
def move_collection(request, collection_id):
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    new_parent_id = data.get("new_parent_id")

    try:
        collection = Collection.objects.get(id=collection_id)
    except Collection.DoesNotExist:
        return JsonResponse({"error": "Collection not found"}, status=404)

    if not collection.user_can_manage(request.user):
        return JsonResponse({"error": "You do not have permission to move this collection"}, status=403)

    new_parent = None
    if new_parent_id:
        try:
            new_parent = Collection.objects.get(id=new_parent_id)
        except Collection.DoesNotExist:
            return JsonResponse({"error": "Target parent collection not found"}, status=404)

    try:
        collection.move_to(new_parent=new_parent)
    except ValidationError as e:
        return JsonResponse({"error": str(e)}, status=400)

    return JsonResponse({
        "message": "Collection moved successfully",
        "collection": {
            "id": collection.id,
            "name": collection.name,
            "parent": collection.parent.id if collection.parent else None,
            "path": collection.get_path(),
        }
    })


@require_http_methods(['GET', 'POST'])
@login_required
def collection_permissions(request, col_id):
    collection = get_object_or_404(Collection, pk=col_id)
    User = get_user_model()

    def serialize_user_for_permissions(user_obj):
        full_name = f"{user_obj.first_name} {user_obj.last_name}".strip()
        return {
            'id': user_obj.id,
            'username': user_obj.username,
            'email': user_obj.email,
            'full_name': full_name if full_name else user_obj.username
        }

    if request.method == 'GET':
        if not collection.user_can_manage(request.user):
            return JsonResponse({'error': 'Permission denied'}, status=403)
        
        view_permissions = CollectionPermission.objects.filter(collection=collection, permission='VIEW').select_related('user')
        edit_permissions = CollectionPermission.objects.filter(collection=collection, permission='EDIT').select_related('user')
        manage_permissions = CollectionPermission.objects.filter(collection=collection, permission='MANAGE').select_related('user')

        viewers = [serialize_user_for_permissions(perm.user) for perm in view_permissions]
        editors = [serialize_user_for_permissions(perm.user) for perm in edit_permissions]
        admins = [serialize_user_for_permissions(perm.user) for perm in manage_permissions]
        
        return JsonResponse({
            'viewers': viewers,
            'editors': editors,
            'admins': admins,
        })

    if not collection.user_can_manage(request.user):
        return JsonResponse({'error': 'Permission denied'}, status=403)

    try:
        data = json.loads(request.body)
        with transaction.atomic():
            CollectionPermission.objects.filter(collection=collection).exclude(user=request.user).delete()

            for viewer in data.get('viewers', []):
                CollectionPermission.objects.create(
                    collection=collection,
                    user=get_user_model().objects.get(id=viewer),
                    permission='VIEW'
                )
            for editor in data.get('editors', []):
                CollectionPermission.objects.create(
                    collection=collection,
                    user=get_user_model().objects.get(id=editor),
                    permission='EDIT'
                )
            for admin in data.get('admins', []):
                CollectionPermission.objects.create(
                    collection=collection,
                    user=get_user_model().objects.get(id=admin),
                    permission='MANAGE'
                )

        return JsonResponse({'status': 'success'})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)


@login_required
@require_http_methods(['GET'])
def collection_detail(request, col_id):
    """Get a single collection with its documents and children."""
    try:
        collection = get_object_or_404(Collection, pk=col_id)
        if not collection.user_can_view(request.user):
            return JsonResponse({'error': 'Permission denied'}, status=403)

        raw_text_type_overrides = _raw_text_type_overrides(collection)
        documents = []
        for model in DESCENDED_FROM_DOCUMENT:
            docs = model.objects.filter(collection=collection)
            for doc in docs:
                model_type = doc.__class__.__name__
                display_type = (
                    raw_text_type_overrides.get(str(doc.id), model_type)
                    if model_type == "RawTextDocument"
                    else model_type
                )
                documents.append({
                    'id': str(doc.id),
                    'title': getattr(doc, 'title', None) or getattr(doc, 'name', 'Untitled'),
                    'type': display_type,
                    'model_type': model_type,
                    'ingestion_date': doc.ingestion_date.isoformat() if hasattr(doc, 'ingestion_date') and doc.ingestion_date else None,
                })

        child_collections = list(collection.children.all())
        document_ids = {doc["id"] for doc in documents}
        document_title_to_id = {}
        for doc in documents:
            normalized_title = " ".join((str(doc.get("title") or "").strip().lower().split()))
            if normalized_title and normalized_title not in document_title_to_id:
                document_title_to_id[normalized_title] = str(doc["id"])
        child_to_parent_doc = _child_collection_parent_document_ids(
            child_collection_ids=[child.id for child in child_collections],
            valid_document_ids=document_ids,
            document_title_to_id=document_title_to_id,
        )
        children = [{
            'id': child.id,
            'name': child.name,
            'document_count': child.document_count(),
            'created_at': child.created_at.isoformat() if hasattr(child, 'created_at') and child.created_at else None,
            'parent_document_id': child_to_parent_doc.get(child.id),
        } for child in child_collections]

        response_data = {
            'collection': {
                'id': collection.id,
                'name': collection.name,
                'path': collection.get_path(),
                'parent': collection.parent.id if collection.parent else None,
                'created_at': collection.created_at.isoformat() if hasattr(collection, 'created_at') and collection.created_at else None,
                'updated_at': collection.updated_at.isoformat() if hasattr(collection, 'updated_at') and collection.updated_at else None,
            },
            'documents': documents,
            'children': children,
            'can_edit': collection.user_can_edit(request.user),
            'can_manage': collection.user_can_manage(request.user),
        }
        return JsonResponse(response_data)
    except Exception as e:
        logger.error(f"Error processing collection data: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)


__all__ = [
    'delete_collection',
    'collections',
    'move_collection',
    'collection_permissions',
    'collection_detail',
]
