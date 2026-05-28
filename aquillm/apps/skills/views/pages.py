"""Page views for the skills feature."""
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.http import Http404, HttpRequest, HttpResponse, HttpResponseForbidden
from django.shortcuts import render
from django.urls import reverse
from django.views.decorators.http import require_http_methods

from apps.collections.models import Collection


@login_required
@require_http_methods(["GET"])
def skills_page(request: HttpRequest) -> HttpResponse:
    """Render the React skills-editor page mount.

    Hidden when SKILLS_ENABLED is off — the feature is unavailable in that case.
    """
    if not getattr(settings, "SKILLS_ENABLED", False):
        raise Http404("Skills feature is disabled")
    return render(request, "aquillm/skills.html")


@login_required
@require_http_methods(["GET"])
def collection_notes_page(request: HttpRequest, collection_id: int) -> HttpResponse:
    """Render the per-collection notes editor page.

    Requires MANAGE on the collection; lower-permission users get 403. The
    React island reads the collection id/name from the mount element's data
    attributes — no extra context fetch on first paint.
    """
    if not getattr(settings, "SKILLS_ENABLED", False):
        raise Http404("Skills feature is disabled")
    try:
        collection = Collection.objects.get(pk=collection_id)
    except Collection.DoesNotExist:
        raise Http404("Collection not found")
    if not collection.user_can_manage(request.user):
        return HttpResponseForbidden("You do not have permission to edit this collection's notes.")
    return render(
        request,
        "aquillm/collection_notes.html",
        {
            "collection_id": collection.id,
            "collection_name": collection.name,
            "collection_url": reverse("collection", args=[collection.id]),
        },
    )


__all__ = ["skills_page", "collection_notes_page"]
