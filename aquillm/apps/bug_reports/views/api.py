"""API views for bug reports."""
import json
import logging
import os

from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods

from apps.bug_reports.models import BugReport
from apps.bug_reports.tracing import get_current_trace_id

logger = logging.getLogger(__name__)

_PAGE_SIZE = 25

_GRAFANA_BASE = os.environ.get("GRAFANA_BASE_URL", "http://localhost:3000")
_TEMPO_DS_UID = os.environ.get("GRAFANA_TEMPO_DS_UID", "tempo")


def _ds():
    return {"type": "tempo", "uid": _TEMPO_DS_UID}


def _grafana_explore_url(panes_dict: dict) -> str:
    """Build a Grafana Explore URL from a panes dict."""
    import json
    import urllib.parse
    panes = json.dumps(panes_dict, separators=(",", ":"))
    return f"{_GRAFANA_BASE}/explore?schemaVersion=1&panes={urllib.parse.quote(panes, safe='')}&orgId=1"


def _grafana_trace_url(trace_id: str) -> str:
    """Build a Grafana Tempo URL for a given trace ID."""
    if not trace_id:
        return ""
    return _grafana_explore_url({"one": {
        "datasource": _TEMPO_DS_UID,
        "queries": [{
            "refId": "A",
            "query": trace_id,
            "queryType": "traceql",
            "datasource": _ds(),
            "editorMode": "code",
        }],
        "range": {"from": "now-1h", "to": "now"},
    }})


def _grafana_user_traces_url(user_id: int, created_at) -> str:
    """Build a Grafana Tempo URL showing the user's recent traces up to the report time."""
    to_ms = int(created_at.timestamp() * 1000)
    return _grafana_explore_url({"one": {
        "datasource": _TEMPO_DS_UID,
        "queries": [{
            "refId": "A",
            "query": f'{{span.enduser.id="{user_id}"}}',
            "queryType": "traceql",
            "datasource": _ds(),
            "editorMode": "code",
            "limit": 50,
        }],
        "range": {"from": str(to_ms - 3600000), "to": str(to_ms)},
    }})


@require_http_methods(["POST"])
@login_required
def submit_bug_report(request):
    """Submit a new bug report. Trace ID is captured from the current span."""
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    title = (data.get("title") or "").strip()
    if not title:
        return JsonResponse({"error": "Title is required"}, status=400)

    report = BugReport.objects.create(
        user=request.user,
        title=title,
        description=(data.get("description") or "").strip(),
        url=(data.get("url") or "")[:2048],
        trace_id=get_current_trace_id(),
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
                "trace_id": r.trace_id,
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
        report = BugReport.objects.select_related("user").get(id=report_id)
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
        "trace_id": report.trace_id,
        "grafana_trace_url": _grafana_trace_url(report.trace_id),
        "grafana_user_traces_url": (
            _grafana_user_traces_url(report.user.id, report.created_at)
            if report.user else ""
        ),
        "created_at": report.created_at.isoformat(),
    }

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
