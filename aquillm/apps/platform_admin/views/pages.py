"""Page views for platform administration."""
import base64
import structlog

from django.contrib.auth.decorators import login_required, user_passes_test
from django.shortcuts import render
from django.views.decorators.http import require_http_methods

from aquillm.ocr_utils import get_gemini_cost_stats
from ..feedbackql.parser import parse
from ..feedbackql.executor import execute
from ..feedbackql.exceptions import FeedbackQLSyntaxError

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
@user_passes_test(lambda u: u.is_superuser)
@require_http_methods(['GET'])
def feedback_dashboard(request):
    """Feedback dashboard — run a query and show results."""
    raw_q = request.GET.get('q', '').strip()

    query_text = ''
    results = []
    columns = []
    error = None

    if raw_q:
        try:
            query_text = base64.b64decode(raw_q.encode()).decode('utf-8')
        except Exception:
            error = 'Could not decode the query parameter. Make sure it is valid base64.'

        if not error:
            try:
                parsed = parse(query_text)
                results = execute(parsed)
                if results:
                    columns = list(results[0].keys())
                    results = [[row.get(col) for col in columns] for row in results]
            except FeedbackQLSyntaxError as exc:
                error = str(exc)
            except Exception as exc:
                logger.exception('feedback_dashboard: unexpected error', exc_info=exc)
                error = 'An unexpected error occurred while running the query.'

    return render(request, 'aquillm/feedback_dashboard.html', {
        'query_text': query_text,
        'results': results,
        'columns': columns,
        'error': error,
        'row_count': len(results),
    })


__all__ = [
    'gemini_cost_monitor',
    'email_whitelist',
    'feedback_dashboard',
]
