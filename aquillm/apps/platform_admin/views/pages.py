"""Page views for platform administration."""
import structlog
from base64 import b64encode
from urllib.parse import quote

from django.contrib.auth.decorators import login_required, user_passes_test
from django.http import HttpResponseRedirect
from django.shortcuts import render
from django.views.decorators.http import require_http_methods

from apps.platform_admin.feedbackql.token_store import resolve_token
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
    """Feedback dashboard — renders the shell; React takes over client-side.

    Two URL forms are accepted for pre-filling a query:
      ?q=<base64>  — self-contained, used by the dashboard's "Copy link" button
      ?t=<token>   — short opaque token minted server-side by the LLM tool;
                     resolved here and re-routed to the ?q= form so React's
                     existing decoder handles it unchanged.
    """
    # Resolve token URLs server-side. Superuser-gating is enforced
    # consistently for both forms by checking before the cache lookup.
    if request.user.is_superuser:
        token = request.GET.get('t')
        if token:
            query = resolve_token(token)
            if query is not None:
                encoded = b64encode(query.encode('utf-8')).decode('ascii')
                return HttpResponseRedirect(
                    f"{request.path}?q={quote(encoded, safe='')}"
                )
            # Token missing/expired — fall through to render with no query.
            # The user lands on the empty dashboard rather than an error.
    return render(request, 'aquillm/feedback_dashboard.html', {
        'access_denied': not request.user.is_superuser,
    })


__all__ = [
    'gemini_cost_monitor',
    'email_whitelist',
    'feedback_dashboard',
]
