"""API views for core app functionality."""
import json
import structlog

from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods

from apps.core.models import UserSettings, COLOR_SCHEME_CHOICES, FONT_FAMILY_CHOICES

logger = structlog.stdlib.get_logger(__name__)


@require_http_methods(["GET", "POST"])
@login_required
def user_settings_api(request):
    """
    GET: return {"color_scheme": ..., "font_family": ...}
    POST: accept JSON {"color_scheme": ..., "font_family": ...},
          validate, save, and return same JSON.
    """
    valid_color_schemes = {key for key, _ in COLOR_SCHEME_CHOICES}
    valid_font_families = {key for key, _ in FONT_FAMILY_CHOICES}

    settings_obj, _ = UserSettings.objects.get_or_create(user=request.user)

    if request.method == "GET":
        return JsonResponse({
            "color_scheme": settings_obj.color_scheme,
            "font_family": settings_obj.font_family,
        })

    # POST
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    color = data.get("color_scheme")
    font = data.get("font_family")

    errors = {}
    if color not in valid_color_schemes:
        errors["color_scheme"] = "Invalid choice"
    if font not in valid_font_families:
        errors["font_family"] = "Invalid choice"
    if errors:
        return JsonResponse({"errors": errors}, status=400)

    settings_obj.color_scheme = color
    settings_obj.font_family = font
    settings_obj.save()

    return JsonResponse({
        "color_scheme": settings_obj.color_scheme,
        "font_family": settings_obj.font_family,
    })


__all__ = [
    'user_settings_api',
]
