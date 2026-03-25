"""API views for bug reports."""
import json
import logging

from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods

from apps.bug_reports.activity import get_activity_log
from apps.bug_reports.models import BugReport, StackTrace

logger = logging.getLogger(__name__)

_PAGE_SIZE = 25


@require_http_methods(["POST"])
@login_required
def submit_bug_report(request):
    """Submit a new bug report. Activity log is snapshotted from Redis."""
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    title = (data.get("title") or "").strip()
    if not title:
        return JsonResponse({"error": "Title is required"}, status=400)

    activity = get_activity_log(request.user.id)

    report = BugReport.objects.create(
        user=request.user,
        title=title,
        description=(data.get("description") or "").strip(),
        url=(data.get("url") or "")[:2048],
        activity_log=activity,
        user_agent=(data.get("user_agent") or "")[:512],
        source="user",
    )
    return JsonResponse({"id": report.id})


def _staff_required(view_func):
    """Decorator that returns 403 for non-staff users."""
    @login_required
    def wrapper(request, *args, **kwargs):
        if not request.user.is_staff:
            return JsonResponse({"error": "Forbidden"}, status=403)
        return view_func(request, *args, **kwargs)
    wrapper.__name__ = view_func.__name__
    wrapper.__module__ = view_func.__module__
    return wrapper


@require_http_methods(["GET"])
@_staff_required
def list_bug_reports(request):
    """List bug reports with optional filtering by source."""
    qs = BugReport.objects.select_related("user")

    source = request.GET.get("source")
    if source in ("user", "exception"):
        qs = qs.filter(source=source)

    page = int(request.GET.get("page", 1))
    offset = (page - 1) * _PAGE_SIZE
    total = qs.count()
    reports = qs[offset:offset + _PAGE_SIZE]

    return JsonResponse({
        "total": total,
        "page": page,
        "page_size": _PAGE_SIZE,
        "results": [
            {
                "id": r.id,
                "title": r.title,
                "user": r.user.username if r.user else None,
                "source": r.source,
                "has_stack_trace": hasattr(r, "stack_trace"),
                "created_at": r.created_at.isoformat(),
            }
            for r in reports
        ],
    })


@require_http_methods(["GET"])
@_staff_required
def bug_report_detail(request, report_id):
    """Return full details for a single bug report."""
    try:
        report = BugReport.objects.select_related("user", "stack_trace").get(id=report_id)
    except BugReport.DoesNotExist:
        return JsonResponse({"error": "Not found"}, status=404)

    data = {
        "id": report.id,
        "title": report.title,
        "description": report.description,
        "url": report.url,
        "user": report.user.username if report.user else None,
        "source": report.source,
        "user_agent": report.user_agent,
        "activity_log": report.activity_log,
        "created_at": report.created_at.isoformat(),
    }

    try:
        st = report.stack_trace
        data["stack_trace"] = {
            "exception_type": st.exception_type,
            "exception_message": st.exception_message,
            "traceback_text": st.traceback_text,
            "request_method": st.request_method,
            "request_path": st.request_path,
            "request_body": st.request_body,
        }
    except StackTrace.DoesNotExist:
        data["stack_trace"] = None

    return JsonResponse(data)


@require_http_methods(["DELETE"])
@_staff_required
def delete_bug_report(request, report_id):
    """Delete a bug report."""
    deleted, _ = BugReport.objects.filter(id=report_id).delete()
    if not deleted:
        return JsonResponse({"error": "Not found"}, status=404)
    return JsonResponse({"ok": True})


def test_exception(request):
    """DEBUG-only view that raises an exception to test the bug report middleware."""
    raise RuntimeError("Intentional test exception for bug report middleware.")
