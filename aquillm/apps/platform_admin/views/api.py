"""API views for platform administration."""
import base64
import structlog

from django.contrib.auth import get_user_model
from django.contrib.auth.decorators import login_required
from django.core.validators import validate_email
from django.core.exceptions import ValidationError
from django.db.models import Q
from django.http import HttpResponseBadRequest, HttpResponseForbidden, JsonResponse, StreamingHttpResponse
from django.utils import timezone
from django.views.decorators.http import require_http_methods

from apps.platform_admin.models import EmailWhitelist
from apps.platform_admin.services.feedback_export import (
    parse_query_bounds,
    stream_feedback_csv_gzip_bytes,
    stream_feedback_csv_lines,
)
from ..feedbackql.parser import (
    parse,
    Condition,
    SelectClause,
    SummarizeClause,
    WhereClause,
)
from ..feedbackql.executor import execute
from ..feedbackql.exceptions import FeedbackQLSyntaxError

# Stable column order used when a row-level query has no select clause.
# Most-useful-for-admins first; technical IDs last.
_DEFAULT_COLUMN_ORDER = [
    'rating',
    'feedback_text',
    'feedback_submitted_at',
    'model',
    'role',
    'content',
    'created_at',
    'tool_call_name',
    'sequence_number',
    'user_id',
    'conversation_id',
    'message_uuid',
]

logger = structlog.stdlib.get_logger(__name__)


def _client_accepts_gzip(request) -> bool:
    """True if Accept-Encoding lists gzip (or x-gzip) with q > 0."""
    for part in request.META.get("HTTP_ACCEPT_ENCODING", "").split(","):
        part = part.strip()
        if not part:
            continue
        pieces = [p.strip() for p in part.split(";")]
        encoding = pieces[0].lower()
        if encoding not in ("gzip", "x-gzip"):
            continue
        q = 1.0
        for p in pieces[1:]:
            if p.lower().startswith("q="):
                try:
                    q = float(p[2:].strip())
                except ValueError:
                    q = 1.0
                break
        if q > 0:
            return True
    return False


@login_required
def search_users(request):
    """Search for users by email, username, or name."""
    query = request.GET.get('query', '').strip()
    exclude_current = request.GET.get('exclude_current', 'false').lower() == 'true'
    
    if not query:
        return JsonResponse({'users': []})

    User = get_user_model()
    users = User.objects.filter(
        Q(email__icontains=query) | 
        Q(username__icontains=query) | 
        Q(first_name__icontains=query) | 
        Q(last_name__icontains=query)
    ).distinct()

    if exclude_current:
        users = users.exclude(id=request.user.id)

    user_list = []
    for user in users:
        full_name = f"{user.first_name} {user.last_name}".strip()
        user_list.append({
            'id': user.id,
            'username': user.username,
            'email': user.email,
            'full_name': full_name
        })
    
    return JsonResponse({'users': user_list})


@login_required
@require_http_methods(['GET'])
def whitelisted_emails(request):
    """Get all whitelisted email addresses."""
    if not request.user.is_staff:
        return JsonResponse({'error': 'Permission denied'}, status=403)
    return JsonResponse({'whitelisted': list(EmailWhitelist.objects.all().values_list('email', flat=True))})


@login_required
@require_http_methods(['POST', 'DELETE'])
def whitelisted_email(request, email):
    """Add or remove a whitelisted email address."""
    if not request.user.is_staff:
        return JsonResponse({'error': 'Permission denied'}, status=403)
    try:
        validate_email(email)
        if request.method == 'POST':
            EmailWhitelist.objects.get_or_create(email=email)
            return JsonResponse({'status': 'success'})
        if request.method == 'DELETE':
            EmailWhitelist.objects.filter(email=email).delete()
            return JsonResponse({'status': 'success'})
    except ValidationError as e:
        return JsonResponse({'error': str(e)}, status=400)
    return JsonResponse({'error': 'Invalid request method'}, status=400)


@login_required
@require_http_methods(["GET"])
def feedback_ratings_csv(request):
    """Stream CSV of message ratings/feedback (superuser only)."""
    if not request.user.is_superuser:
        return HttpResponseForbidden("Superuser access required")

    start_d, end_d = parse_query_bounds(
        request.GET.get("start_date"),
        request.GET.get("end_date"),
    )

    min_rating = None
    raw_min = request.GET.get("min_rating")
    if raw_min not in (None, ""):
        try:
            min_rating = int(raw_min)
        except ValueError:
            return HttpResponseBadRequest("Invalid min_rating")

    user_number = None
    raw_user = request.GET.get("user_number")
    if raw_user not in (None, ""):
        try:
            user_number = int(raw_user)
        except ValueError:
            return HttpResponseBadRequest("Invalid user_number")

    def content_plain():
        yield from stream_feedback_csv_lines(
            start_date=start_d,
            end_date=end_d,
            min_rating=min_rating,
            user_number=user_number,
        )

    if _client_accepts_gzip(request):
        response = StreamingHttpResponse(
            stream_feedback_csv_gzip_bytes(
                start_date=start_d,
                end_date=end_d,
                min_rating=min_rating,
                user_number=user_number,
            ),
            content_type="text/csv; charset=utf-8",
        )
        response["Content-Encoding"] = "gzip"
        response["Vary"] = "Accept-Encoding"
    else:
        response = StreamingHttpResponse(content_plain(), content_type="text/csv; charset=utf-8")

    fname = f'feedback_ratings_{timezone.now().strftime("%Y%m%d")}.csv'
    response["Content-Disposition"] = f'attachment; filename="{fname}"'
    return response


@login_required
@require_http_methods(['GET'])
def feedback_dashboard_query(request):
    """Run a FeedbackQL query and return structured JSON results."""
    if not request.user.is_superuser:
        return JsonResponse({'error': 'forbidden'}, status=403)

    raw_q = request.GET.get('q', '').strip()
    if not raw_q:
        return JsonResponse({
            'query_text': '',
            'rows': [],
            'columns': [],
            'is_row_level': True,
            'chart_data': None,
            'row_count': 0,
        })

    try:
        query_text = base64.b64decode(raw_q.encode()).decode('utf-8')
    except Exception:
        return JsonResponse(
            {'error': 'Could not decode the query parameter. Make sure it is valid base64.'},
            status=400,
        )

    try:
        parsed = parse(query_text)
        notice = _detect_query_tips(parsed)
        results_dicts = execute(parsed)
    except FeedbackQLSyntaxError as exc:
        return JsonResponse({'query_text': query_text, 'error': str(exc)}, status=400)
    except Exception as exc:
        logger.exception('feedback_dashboard_query: unexpected error', exc_info=exc)
        return JsonResponse(
            {'query_text': query_text, 'error': 'An unexpected error occurred while running the query.'},
            status=500,
        )

    summarize = next((c for c in parsed.clauses if isinstance(c, SummarizeClause)), None)
    is_row_level = summarize is None

    columns: list[str] = []
    rows: list[dict] = []
    chart_data = None

    if results_dicts:
        if is_row_level:
            select_clause = next((c for c in parsed.clauses if isinstance(c, SelectClause)), None)
            present = set(results_dicts[0].keys())
            if select_clause:
                # User chose specific fields (and ordered them) — respect that order.
                # Metadata fields like conversation_id / message_uuid stay in the row
                # data for the thread viewer but are only shown as columns if the user
                # explicitly listed them.
                columns = [f for f in select_clause.fields if f in present]
            else:
                # No select clause — order columns most-useful to least-useful for
                # stable, predictable display across queries.
                columns = [f for f in _DEFAULT_COLUMN_ORDER if f in present]
                # Append any extras not in the preferred order (defensive).
                columns += [k for k in results_dicts[0].keys() if k not in columns]
            rows = [
                {
                    'cells': [_to_jsonable(row.get(col)) for col in columns],
                    'conversation_id': str(row.get('conversation_id', '')),
                    'message_uuid': str(row.get('message_uuid', '')),
                }
                for row in results_dicts
            ]
        else:
            columns = list(results_dicts[0].keys())
            rows = [
                {'cells': [_to_jsonable(row.get(col)) for col in columns]}
                for row in results_dicts
            ]

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
            chart_data = {'labels': labels, 'datasets': datasets}

    return JsonResponse({
        'query_text': query_text,
        'rows': rows,
        'columns': columns,
        'is_row_level': is_row_level,
        'chart_data': chart_data,
        'row_count': len(rows),
        'notice': notice,
    })


def _iter_conditions(clause):
    """Yield every Condition reachable from a parsed clause's parts list."""
    parts = getattr(clause, 'parts', None)
    if not parts:
        return
    for part in parts:
        if isinstance(part, Condition):
            yield part


def _detect_query_tips(parsed):
    """
    Return a friendly tip string when the query has a likely-wrong but
    syntactically-valid pattern. Returns None if there's nothing to flag.

    Currently detects: `tools_used == "<tool>"` on the conversations stream.
    `tools_used` is a comma-separated string, so == only matches conversations
    that used *exactly* that one tool — silently dropping any conversation
    that combined it with another tool. Suggest `contains` instead.
    """
    if parsed.stream != 'conversations':
        return None
    for clause in parsed.clauses:
        if not isinstance(clause, WhereClause):
            continue
        for cond in _iter_conditions(clause):
            if (cond.field == 'tools_used'
                    and cond.op == '=='
                    and isinstance(cond.value, str)):
                return (
                    f'Tip: tools_used is a comma-separated list of every tool '
                    f'a conversation used, so "tools_used == \"{cond.value}\"" '
                    f'only matches conversations that used exactly that one '
                    f'tool — multi-tool conversations are silently dropped. '
                    f'For "any conversation that used {cond.value}", use '
                    f'"tools_used contains \"{cond.value}\"" instead.'
                )
    return None


def _to_jsonable(value):
    """Coerce DB values (datetimes, UUIDs, etc.) into JSON-safe primitives."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


@login_required
@require_http_methods(['GET'])
def feedback_dashboard_conversation(request):
    """Return all messages in a conversation for the thread viewer modal."""
    from apps.chat.models import Message

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
        logger.exception('feedback_dashboard_conversation: unexpected error', exc_info=exc)
        return JsonResponse({'error': 'An unexpected error occurred.'}, status=500)


__all__ = [
    'feedback_ratings_csv',
    'search_users',
    'whitelisted_emails',
    'whitelisted_email',
    'feedback_dashboard_query',
    'feedback_dashboard_conversation',
]
