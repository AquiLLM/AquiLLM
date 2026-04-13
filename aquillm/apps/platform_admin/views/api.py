import csv
import io
import json
import logging
import math
import zlib
from datetime import datetime, timezone as py_tz

from django.contrib.auth import get_user_model
from django.contrib.auth.decorators import login_required
from django.core.validators import validate_email
from django.core.exceptions import ValidationError
from django.db.models import Q
from django.http import (
    HttpResponseBadRequest,
    HttpResponseForbidden,
    JsonResponse,
    StreamingHttpResponse,
)
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
from apps.platform_admin.services.feedback_aggregates import (
    get_summary_metrics,
    get_filter_options,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------

def _require_superuser(request):
    """
    return a 403 response if user is not a superuser, else return none,
    call this at the top of every dashboard endpoint
    """
    if not request.user.is_superuser:
        return HttpResponseForbidden("superuser access required")
    return None


def _client_accepts_gzip(request) -> bool:
    """true if accept-encoding lists gzip with q > 0"""
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


def _to_iso(dt) -> str | None:
    """convert a datetime to iso utc string, returns none if input is none"""
    if dt is None:
        return None
    if hasattr(dt, "tzinfo") and dt.tzinfo is None:
        dt = dt.replace(tzinfo=py_tz.utc)
    return dt.astimezone(py_tz.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _serialize_row(msg) -> dict:
    """
    convert a Message orm object to a json-safe dict for the rows api,
    uses the annotated fields from feedback_dataset_queryset so no
    extra queries are needed per row
    """
    return {
        "id": msg.id,
        "message_uuid": str(msg.message_uuid),
        "conversation_id": msg.conversation_id,
        "conversation_name": msg.conversation_name,
        "user_id": msg.user_id,
        "username": msg.username,
        "rating": msg.rating,
        "feedback_text": msg.feedback_text,
        "feedback_submitted_at": _to_iso(msg.feedback_submitted_at),
        "created_at": _to_iso(msg.created_at),
        "effective_date": _to_iso(msg.effective_date),
        "role": msg.role,
        "content_snippet": msg.content_snippet,
        "model": msg.model,
        "tool_call_name": msg.tool_call_name,
        "usage": msg.usage,
        "has_feedback_text": msg.has_feedback_text,
    }

@login_required
def search_users(request):
    """search for users by email, username, or name"""
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
            'full_name': full_name,
        })

    return JsonResponse({'users': user_list})


@login_required
@require_http_methods(['GET'])
def whitelisted_emails(request):
    """get all whitelisted email addresses"""
    if not request.user.is_staff:
        return JsonResponse({'error': 'Permission denied'}, status=403)
    return JsonResponse({
        'whitelisted': list(
            EmailWhitelist.objects.all().values_list('email', flat=True)
        )
    })


@login_required
@require_http_methods(['POST', 'DELETE'])
def whitelisted_email(request, email):
    """add or remove a whitelisted email address"""
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
    """stream csv of message ratings and feedback, superuser only"""
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
        response = StreamingHttpResponse(
            content_plain(), content_type="text/csv; charset=utf-8"
        )

    fname = f'feedback_ratings_{timezone.now().strftime("%Y%m%d")}.csv'
    response["Content-Disposition"] = f'attachment; filename="{fname}"'
    return response


# ---------------------------------------------------------------------------
# dashboard api endpoints — all superuser only
# ---------------------------------------------------------------------------

@login_required
@require_http_methods(["GET"])
def feedback_dashboard_rows(request):
    """
    return paginated feedback rows for the dashboard table

    query params accepted:
        all FeedbackFilters fields from feedback_dataset.py
        page        int, default 1
        page_size   int, default 50, capped at 200

    response shape:
        rows        list of serialized message dicts
        page        current page number
        page_size   rows per page
        total_count total rows matching filters
        total_pages total pages at this page_size
    """
    denied = _require_superuser(request)
    if denied:
        return denied

    try:
        page = max(1, int(request.GET.get("page", 1)))
    except (ValueError, TypeError):
        page = 1

    try:
        page_size = min(200, max(1, int(request.GET.get("page_size", 50))))
    except (ValueError, TypeError):
        page_size = 50

    filters = FeedbackFilters.from_request_params(request.GET.dict())

    try:
        qs = get_filtered_queryset(filters)
        total_count = qs.count()
        offset = (page - 1) * page_size
        rows = list(qs[offset: offset + page_size])
        total_pages = math.ceil(total_count / page_size) if total_count > 0 else 1

        return JsonResponse({
            "rows": [_serialize_row(r) for r in rows],
            "page": page,
            "page_size": page_size,
            "total_count": total_count,
            "total_pages": total_pages,
        })
    except Exception as exc:
        logger.exception("error in feedback_dashboard_rows: %s", exc)
        return JsonResponse({"error": "internal server error"}, status=500)


@login_required
@require_http_methods(["GET"])
def feedback_dashboard_summary(request):
    """
    return aggregate summary metrics for the dashboard header cards

    query params accepted:
        all FeedbackFilters fields

    response shape:
        total_count         int
        rated_count         int
        avg_rating          float or null
        rating_distribution dict mapping str rating to count
        has_text_count      int
        date_min            iso utc string or null
        date_max            iso utc string or null
    """
    denied = _require_superuser(request)
    if denied:
        return denied

    filters = FeedbackFilters.from_request_params(request.GET.dict())

    try:
        summary = get_summary_metrics(filters)
        # convert int keys in rating_distribution to strings for json
        summary["rating_distribution"] = {
            str(k): v for k, v in summary["rating_distribution"].items()
        }
        return JsonResponse(summary)
    except Exception as exc:
        logger.exception("error in feedback_dashboard_summary: %s", exc)
        return JsonResponse({"error": "internal server error"}, status=500)


@login_required
@require_http_methods(["GET"])
def feedback_dashboard_filters(request):
    """
    return available filter option values for populating ui dropdowns

    no query params needed, always returns the full universe of options

    response shape:
        users       list of dicts with id and username
        roles       list of strings
        models      list of strings
        tool_names  list of strings
        ratings     list of ints
    """
    denied = _require_superuser(request)
    if denied:
        return denied

    filters = FeedbackFilters.from_request_params(request.GET.dict())

    try:
        options = get_filter_options()
        return JsonResponse(options)
    except Exception as exc:
        logger.exception("error in feedback_dashboard_filters: %s", exc)
        return JsonResponse({"error": "internal server error"}, status=500)


@login_required
@require_http_methods(["GET"])
def feedback_dashboard_export(request):
    """
    stream a csv export of feedback rows matching current filters,
    supports gzip compression via accept-encoding,
    uses the same filter logic as feedback_dashboard_rows so the
    export always matches what the ui is showing

    query params accepted:
        all FeedbackFilters fields
    """
    denied = _require_superuser(request)
    if denied:
        return denied

    filters = FeedbackFilters.from_request_params(request.GET.dict())

    def _iter_csv_lines():
        # write the header row first
        header_buf = io.StringIO()
        csv.writer(
            header_buf, quoting=csv.QUOTE_MINIMAL, lineterminator="\n"
        ).writerow([
            "date",
            "username",
            "user_id",
            "conversation_name",
            "role",
            "rating",
            "feedback_text",
            "model",
            "tool_call_name",
            "content_snippet",
        ])
        yield header_buf.getvalue()

        # use the same filtered queryset as the rows endpoint
        # iterate in chunks to avoid loading the full result into memory
        qs = get_filtered_queryset(filters)
        for msg in qs.iterator(chunk_size=500):
            line_buf = io.StringIO()
            csv.writer(
                line_buf, quoting=csv.QUOTE_MINIMAL, lineterminator="\n"
            ).writerow([
                _to_iso(msg.feedback_submitted_at or msg.created_at),
                msg.username,
                msg.user_id,
                msg.conversation_name or "",
                msg.role,
                "" if msg.rating is None else msg.rating,
                msg.feedback_text or "",
                msg.model or "",
                msg.tool_call_name or "",
                msg.content_snippet or "",
            ])
            yield line_buf.getvalue()

    if _client_accepts_gzip(request):
        def _gzip_gen():
            compressor = zlib.compressobj(9, zlib.DEFLATED, zlib.MAX_WBITS | 16)
            for text in _iter_csv_lines():
                chunk = compressor.compress(text.encode("utf-8"))
                if chunk:
                    yield chunk
            tail = compressor.flush()
            if tail:
                yield tail

        response = StreamingHttpResponse(
            _gzip_gen(), content_type="text/csv; charset=utf-8"
        )
        response["Content-Encoding"] = "gzip"
        response["Vary"] = "Accept-Encoding"
    else:
        response = StreamingHttpResponse(
            _iter_csv_lines(), content_type="text/csv; charset=utf-8"
        )

    fname = f'feedback_dashboard_{timezone.now().strftime("%Y%m%d_%H%M%S")}.csv'
    response["Content-Disposition"] = f'attachment; filename="{fname}"'
    return response


__all__ = [
    'feedback_ratings_csv',
    'feedback_dashboard_rows',
    'feedback_dashboard_summary',
    'feedback_dashboard_filters',
    'feedback_dashboard_export',
    'search_users',
    'whitelisted_emails',
    'whitelisted_email',
]