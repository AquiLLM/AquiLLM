"""JSON CRUD endpoint for per-collection markdown notes."""
from __future__ import annotations

import json

import structlog
from django.contrib.auth.decorators import login_required
from django.http import HttpRequest, JsonResponse
from django.views.decorators.http import require_http_methods

from apps.collections.models import Collection
from apps.notes.models import CollectionNote
from apps.notes.models.collection_note import MAX_BODY_LENGTH as COLLECTION_NOTE_MAX_BODY

logger = structlog.stdlib.get_logger(__name__)


def _parse_payload(request: HttpRequest) -> dict | JsonResponse:
    try:
        data = json.loads(request.body or b"{}")
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    if not isinstance(data, dict):
        return JsonResponse({"error": "Payload must be a JSON object"}, status=400)
    return data


def _serialize_collection_note(note: CollectionNote | None, collection: Collection) -> dict:
    if note is None:
        return {
            "collection_id": collection.id,
            "collection_name": collection.name,
            "body": "",
            "updated_at": None,
            "updated_by": None,
            "exists": False,
            "max_body_length": COLLECTION_NOTE_MAX_BODY,
        }
    return {
        "collection_id": collection.id,
        "collection_name": collection.name,
        "body": note.body,
        "updated_at": note.updated_at.isoformat() if note.updated_at else None,
        "updated_by": note.updated_by.username if note.updated_by else None,
        "exists": True,
        "max_body_length": COLLECTION_NOTE_MAX_BODY,
    }


@login_required
@require_http_methods(["GET", "PUT"])
def collection_note_detail(request: HttpRequest, collection_id: int) -> JsonResponse:
    """Read or upsert the Collection Notes for a single collection.

    Permission rules:
      - GET requires EDIT or higher on the collection.
      - PUT requires MANAGE on the collection.
    PUT is upsert: creates the row if it does not exist, updates otherwise.
    """
    try:
        collection = Collection.objects.get(pk=collection_id)
    except Collection.DoesNotExist:
        return JsonResponse({"error": "Collection not found"}, status=404)

    if request.method == "GET":
        if not collection.user_can_edit(request.user):
            return JsonResponse({"error": "Forbidden"}, status=403)
        note = CollectionNote.objects.filter(collection=collection).first()
        return JsonResponse(_serialize_collection_note(note, collection))

    # PUT
    if not collection.user_can_manage(request.user):
        return JsonResponse({"error": "Forbidden"}, status=403)
    payload = _parse_payload(request)
    if isinstance(payload, JsonResponse):
        return payload
    body = payload.get("body", "")
    if not isinstance(body, str):
        return JsonResponse({"error": "body must be a string"}, status=400)
    if len(body) > COLLECTION_NOTE_MAX_BODY:
        return JsonResponse(
            {"error": f"body too long (max {COLLECTION_NOTE_MAX_BODY} chars)"}, status=400
        )
    note, _ = CollectionNote.objects.update_or_create(
        collection=collection,
        defaults={"body": body, "updated_by": request.user},
    )
    return JsonResponse(_serialize_collection_note(note, collection))


__all__ = ["collection_note_detail"]
