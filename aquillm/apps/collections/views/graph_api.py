from __future__ import annotations

from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from django.views.decorators.http import require_http_methods

from apps.collections.models import Collection
from apps.collections.services.graph_visualization import collection_graph_envelope
from apps.knowledge_graph.services.builds import (
    RebuildPublicationError,
    create_rebuild_request,
)

from .schema_api_helpers import require_edit, require_view


def _collection(col_id: int) -> Collection:
    return get_object_or_404(Collection, id=col_id)


@login_required
@require_http_methods(["GET"])
def graph_visualization(request, col_id: int):
    collection = _collection(col_id)
    if denied := require_view(collection, request.user):
        return denied
    query = request.GET.get("q", "").strip()
    if len(query) > 128:
        return JsonResponse({"error": "query_too_long"}, status=400)
    return JsonResponse(
        collection_graph_envelope(collection, request.user, query=query)
    )


@login_required
@require_http_methods(["POST"])
def graph_rebuild(request, col_id: int):
    collection = _collection(col_id)
    if denied := require_edit(collection, request.user):
        return denied
    try:
        rebuild = create_rebuild_request(
            scope_type="collection",
            scope_id=collection.pk,
        )
    except RebuildPublicationError as exc:
        return JsonResponse(
            {"error": exc.error_code, "request_id": str(exc.request_id)},
            status=503,
        )
    return JsonResponse(
        {"request_id": str(rebuild.pk), "status": rebuild.status},
        status=202,
    )


__all__ = ["graph_rebuild", "graph_visualization"]
