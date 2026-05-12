"""Page views for the skills feature."""
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.http import Http404, HttpRequest, HttpResponse
from django.shortcuts import render
from django.views.decorators.http import require_http_methods


@login_required
@require_http_methods(["GET"])
def skills_page(request: HttpRequest) -> HttpResponse:
    """Render the React skills-editor page mount.

    Hidden when SKILLS_ENABLED is off — the feature is unavailable in that case.
    """
    if not getattr(settings, "SKILLS_ENABLED", False):
        raise Http404("Skills feature is disabled")
    return render(request, "aquillm/skills.html")


__all__ = ["skills_page"]
