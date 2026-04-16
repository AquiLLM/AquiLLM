"""Page views for platform administration."""
import base64
import json
import structlog

from django.contrib.auth.decorators import login_required, user_passes_test
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_http_methods

from aquillm.ocr_utils import get_gemini_cost_stats
from apps.chat.models import Message
from ..feedbackql.parser import parse, SummarizeClause
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
@require_http_methods(['GET'])
def feedback_dashboard(request):
    """Feedback dashboard — run a query and show results."""
    if not request.user.is_superuser:
        return render(request, 'aquillm/feedback_dashboard.html', {'access_denied': True})

    raw_q = request.GET.get('q', '').strip()

    query_text = ''
    columns = []
    error = None
    chart_json = None

    # rows_for_template: list of dicts with 'cells' (display values) and
    # optional 'conversation_id' / 'message_uuid' for the thread viewer.
    rows_for_template = []
    # True when results are individual message rows (not an aggregate summary).
    is_row_level = False

    if raw_q:
        try:
            query_text = base64.b64decode(raw_q.encode()).decode('utf-8')
        except Exception:
            error = 'Could not decode the query parameter. Make sure it is valid base64.'

        if not error:
            try:
                parsed = parse(query_text)
                results_dicts = execute(parsed)

                summarize = next(
                    (c for c in parsed.clauses if isinstance(c, SummarizeClause)), None
                )
                is_row_level = summarize is None

                if results_dicts:
                    if is_row_level:
                        # Executor always includes conversation_id and message_uuid —
                        # strip them from the display columns but keep as row metadata.
                        meta = {'conversation_id', 'message_uuid'}
                        columns = [k for k in results_dicts[0].keys() if k not in meta]
                        rows_for_template = [
                            {
                                'cells': [row.get(col) for col in columns],
                                'conversation_id': str(row.get('conversation_id', '')),
                                'message_uuid': str(row.get('message_uuid', '')),
                            }
                            for row in results_dicts
                        ]
                    else:
                        columns = list(results_dicts[0].keys())
                        rows_for_template = [
                            {'cells': [row.get(col) for col in columns]}
                            for row in results_dicts
                        ]

                    # Build chart data when the query groups by one or more fields
                    if summarize and summarize.by:
                        by_field = summarize.by[0]
                        agg_aliases = [agg.alias for agg in summarize.aggregations]
                        labels = [
                            str(r.get(by_field)) if r.get(by_field) is not None else '(none)'
                            for r in results_dicts
                        ]
                        datasets = [
                            {'label': alias, 'data': [r.get(alias) for r in results_dicts]}
                            for alias in agg_aliases
                        ]
                        chart_json = json.dumps({'labels': labels, 'datasets': datasets})

            except FeedbackQLSyntaxError as exc:
                error = str(exc)
            except Exception as exc:
                logger.exception('feedback_dashboard: unexpected error', exc_info=exc)
                error = 'An unexpected error occurred while running the query.'

    return render(request, 'aquillm/feedback_dashboard.html', {
        'query_text': query_text,
        'rows': rows_for_template,
        'columns': columns,
        'error': error,
        'row_count': len(rows_for_template),
        'chart_json': chart_json,
        'is_row_level': is_row_level,
    })


@login_required
@require_http_methods(['GET'])
def conversation_thread(request):
    """Return all messages in a conversation as JSON, for the thread viewer modal."""
    if not request.user.is_superuser:
        return JsonResponse({'error': 'forbidden'}, status=403)

    conv_id = request.GET.get('id', '').strip()
    if not conv_id:
        return JsonResponse({'error': 'missing id'}, status=400)

    try:
        messages = (
            Message.objects
            .filter(conversation_id=conv_id)
            .order_by('sequence_number')
            .values(
                'message_uuid', 'role', 'content', 'model',
                'sequence_number', 'created_at',
                'rating', 'feedback_text',
                'tool_call_name', 'tool_call_input',
                'tool_name', 'result_dict', 'for_whom',
            )
        )
        data = []
        for m in messages:
            data.append({
                'message_uuid': str(m['message_uuid']),
                'role': m['role'],
                'content': m['content'] or '',
                'model': m['model'],
                'sequence_number': m['sequence_number'],
                'created_at': m['created_at'].isoformat() if m['created_at'] else None,
                'rating': m['rating'],
                'feedback_text': m['feedback_text'],
                'tool_call_name': m['tool_call_name'],
                'tool_call_input': m['tool_call_input'],
                'tool_name': m['tool_name'],
                'result_dict': m['result_dict'],
                'for_whom': m['for_whom'],
            })
        return JsonResponse({'messages': data})
    except Exception as exc:
        logger.exception('conversation_thread: unexpected error', exc_info=exc)
        return JsonResponse({'error': 'An unexpected error occurred.'}, status=500)


__all__ = [
    'gemini_cost_monitor',
    'email_whitelist',
    'feedback_dashboard',
    'conversation_thread',
]
