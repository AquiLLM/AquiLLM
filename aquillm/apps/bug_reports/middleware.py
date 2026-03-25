"""HTTP middleware for activity logging and exception capture."""
import logging
import traceback

from .activity import get_activity_log, log_activity, now_iso

logger = logging.getLogger(__name__)

_SKIP_PREFIXES = ('/static/', '/health', '/ready', '/favicon.ico')

_SENSITIVE_KEYS = frozenset({
    'password', 'passwd', 'secret', 'token', 'api_key', 'apikey',
    'authorization', 'session', 'csrfmiddlewaretoken',
})

_MAX_BODY_SIZE = 10_240  # 10 KB


def _sanitize_body(request) -> str:
    """Return a sanitized, truncated copy of the request body."""
    content_type = request.content_type or ''
    if not content_type.startswith(('application/json', 'application/x-www-form-urlencoded', 'text/')):
        return f"[binary: {content_type}]"
    try:
        body = request.body.decode('utf-8', errors='replace')[:_MAX_BODY_SIZE]
    except Exception:
        return '[unreadable]'
    for key in _SENSITIVE_KEYS:
        # Crude redaction — covers JSON "key":"value" and form key=value
        import re
        body = re.sub(
            rf'(["\']?{re.escape(key)}["\']?\s*[:=]\s*)["\']?[^"\'&,\s]+["\']?',
            rf'\1"[REDACTED]"',
            body,
            flags=re.IGNORECASE,
        )
    return body


class BugReportMiddleware:
    """Logs HTTP requests to Redis and auto-creates bug reports from exceptions."""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # Log the request to Redis for authenticated users
        if (
            request.user
            and getattr(request.user, 'is_authenticated', False)
            and not any(request.path.startswith(p) for p in _SKIP_PREFIXES)
        ):
            log_activity(request.user.id, {
                'type': 'http_request',
                'method': request.method,
                'path': request.path,
                'timestamp': now_iso(),
            })

        return self.get_response(request)

    def process_exception(self, request, exception):
        """Auto-create a BugReport + StackTrace from unhandled exceptions."""
        try:
            from .models import BugReport, StackTrace

            user = None
            activity = []
            if request.user and getattr(request.user, 'is_authenticated', False):
                user = request.user
                activity = get_activity_log(user.id)

            report = BugReport.objects.create(
                user=user,
                title=f"{type(exception).__name__}: {str(exception)[:200]}",
                description='Auto-generated from unhandled exception.',
                url=request.build_absolute_uri(),
                activity_log=activity,
                user_agent=request.META.get('HTTP_USER_AGENT', '')[:512],
                source='exception',
            )
            StackTrace.objects.create(
                bug_report=report,
                exception_type=type(exception).__name__,
                exception_message=str(exception),
                traceback_text=''.join(traceback.format_exception(exception)),
                request_method=request.method,
                request_path=request.path,
                request_body=_sanitize_body(request),
            )
        except Exception:
            logger.debug("Failed to create bug report from exception", exc_info=True)

        return None
