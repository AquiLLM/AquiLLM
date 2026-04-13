import logging

from django.contrib.auth.decorators import login_required, user_passes_test
from django.http import HttpResponseForbidden
from django.shortcuts import render
from django.views.decorators.http import require_http_methods

from aquillm.ocr_utils import get_gemini_cost_stats

logger = logging.getLogger(__name__)


@login_required
@require_http_methods(['GET'])
def gemini_cost_monitor(request):
    stats = get_gemini_cost_stats()
    return render(request, 'aquillm/gemini_cost_monitor.html', {'stats': stats})


@login_required
@user_passes_test(lambda u: u.is_staff)
@require_http_methods(['GET'])
def email_whitelist(request):
    return render(request, 'aquillm/email_whitelist.html')


@login_required
@require_http_methods(['GET'])
def feedback_dashboard(request):
    """
    superuser-only feedback analytics dashboard page

    non-superusers get a 403 even if they know the url,
    the sidebar link is hidden from them in base.html but
    this explicit check is the real security gate
    """
    if not request.user.is_superuser:
        return HttpResponseForbidden(
            "you do not have permission to access the feedback dashboard"
        )
    return render(request, 'aquillm/feedback_dashboard.html')


__all__ = [
    'gemini_cost_monitor',
    'email_whitelist',
    'feedback_dashboard',
]