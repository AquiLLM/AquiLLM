"""Page views for platform administration."""
import structlog

from django.contrib.auth.decorators import login_required, user_passes_test
from django.http import HttpResponseForbidden
from django.shortcuts import render
from django.views.decorators.http import require_http_methods

from aquillm.ocr_utils import get_gemini_cost_stats

logger = structlog.stdlib.get_logger(__name__)


@login_required
@require_http_methods(['GET'])
def gemini_cost_monitor(request):
    """View to display the current Gemini API cost statistics."""
    stats = get_gemini_cost_stats()
    return render(request, 'aquillm/gemini_cost_monitor.html', {'stats': stats})


@login_required
@user_passes_test(lambda u: u.is_staff)
@require_http_methods(['GET'])
def email_whitelist(request):
    """Display the email whitelist management page."""
    return render(request, 'aquillm/email_whitelist.html')


@login_required
@require_http_methods(['GET'])
def feedback_dashboard(request):
    """Display the superuser-only feedback dashboard page."""
    if not request.user.is_superuser:
        return HttpResponseForbidden("Superuser access required")

    return render(request, 'aquillm/feedback_dashboard.html')


__all__ = [
    'feedback_dashboard',
    'gemini_cost_monitor',
    'email_whitelist',
]
