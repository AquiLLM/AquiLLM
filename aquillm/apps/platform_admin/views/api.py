"""API views for platform administration."""
import math
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
from apps.platform_admin.services.feedback_dataset import (
    FeedbackFilters,
    get_filtered_queryset,
)
from apps.platform_admin.services.feedback_prql import build_feedback_prql

logger = structlog.stdlib.get_logger(__name__)


def _require_superuser(request):
    """Return a 403 response when the current user is not a superuser."""
    if not request.user.is_superuser:
        return HttpResponseForbidden("Superuser access required")
    return None


def _to_iso_utc(dt):
    """Convert a datetime to an ISO-8601 UTC string."""
    if dt is None:
        return None
    if timezone.is_naive(dt):
        dt = timezone.make_aware(dt, timezone.get_current_timezone())
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_positive_int(value, *, default: int, maximum: int | None = None) -> int:
    """Parse a positive integer query parameter with a default and optional cap."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default

    parsed = max(1, parsed)
    if maximum is not None:
        parsed = min(maximum, parsed)
    return parsed


def _serialize_feedback_row(message) -> dict:
    """Serialize an annotated Message row for the dashboard table."""
    return {
        "id": message.id,
        "message_uuid": str(message.message_uuid),
        "conversation_id": message.conversation_id,
        "conversation_name": message.conversation_name,
        "user_id": message.user_id,
        "username": message.username,
        "rating": message.rating,
        "feedback_text": message.feedback_text,
        "feedback_submitted_at": _to_iso_utc(message.feedback_submitted_at),
        "created_at": _to_iso_utc(message.created_at),
        "effective_date": _to_iso_utc(message.effective_date),
        "role": message.role,
        "content_snippet": message.content_snippet,
        "model": message.model,
        "tool_call_name": message.tool_call_name,
        "usage": message.usage,
        "has_feedback_text": message.has_feedback_text,
    }



def _feedback_filters_to_prql_specs(filters: FeedbackFilters) -> list[dict]:
    """Convert parsed dashboard filters into display-only PRQL filter specs."""
    specs = []

    if filters.start_date:
        specs.append({
            "field": "effective_date",
            "operator": "on_or_after",
            "value": filters.start_date,
        })

    if filters.end_date:
        specs.append({
            "field": "effective_date",
            "operator": "on_or_before",
            "value": filters.end_date,
        })

    if filters.user_id is not None:
        specs.append({
            "field": "user_id",
            "operator": "equals",
            "value": filters.user_id,
        })

    if filters.exact_rating is not None:
        specs.append({
            "field": "rating",
            "operator": "equals",
            "value": filters.exact_rating,
        })
    else:
        if filters.min_rating is not None:
            specs.append({
                "field": "rating",
                "operator": "greater_than_or_equal",
                "value": filters.min_rating,
            })
        if filters.max_rating is not None:
            specs.append({
                "field": "rating",
                "operator": "less_than_or_equal",
                "value": filters.max_rating,
            })

    if filters.feedback_text_search:
        specs.append({
            "field": "feedback_text",
            "operator": "contains",
            "value": filters.feedback_text_search,
        })

    if filters.conversation_name_search:
        specs.append({
            "field": "conversation_name",
            "operator": "contains",
            "value": filters.conversation_name_search,
        })

    if filters.role:
        specs.append({
            "field": "role",
            "operator": "equals",
            "value": filters.role,
        })

    if filters.model:
        specs.append({
            "field": "model",
            "operator": "equals",
            "value": filters.model,
        })

    if filters.tool_call_name:
        specs.append({
            "field": "tool_call_name",
            "operator": "equals",
            "value": filters.tool_call_name,
        })

    if filters.has_feedback_text is not None:
        specs.append({
            "field": "has_feedback_text",
            "operator": "equals",
            "value": filters.has_feedback_text,
        })

    return specs


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
@require_http_methods(["GET"])
def feedback_dashboard_rows(request):
    """Return paginated feedback rows for the dashboard table."""
    denied = _require_superuser(request)
    if denied:
        return denied

    page = _parse_positive_int(request.GET.get("page"), default=1)
    page_size = _parse_positive_int(
        request.GET.get("page_size"),
        default=50,
        maximum=200,
    )

    filters = FeedbackFilters.from_request_params(request.GET.dict())
    queryset = get_filtered_queryset(filters)

    total_count = queryset.count()
    offset = (page - 1) * page_size
    rows = queryset[offset:offset + page_size]
    total_pages = math.ceil(total_count / page_size) if total_count else 1

    prql = build_feedback_prql(_feedback_filters_to_prql_specs(filters))

    return JsonResponse({
        "rows": [_serialize_feedback_row(row) for row in rows],
        "page": page,
        "page_size": page_size,
        "total_count": total_count,
        "total_pages": total_pages,
        "prql": prql,
    })


__all__ = [
    'feedback_dashboard_rows',
    'feedback_ratings_csv',
    'search_users',
    'whitelisted_emails',
    'whitelisted_email',
]
