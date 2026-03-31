"""HTTP middleware for exception capture and user trace tagging."""
import structlog
import traceback

from .tracing import get_current_trace_id, set_user_attribute

logger = structlog.stdlib.get_logger(__name__)

_SKIP_PREFIXES = ('/static/', '/health', '/ready', '/favicon.ico')


class BugReportMiddleware:
    """Tags traces with user ID and auto-creates bug reports from unhandled exceptions."""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if (
            request.user
            and getattr(request.user, 'is_authenticated', False)
            and not any(request.path.startswith(p) for p in _SKIP_PREFIXES)
        ):
            set_user_attribute(request.user.id)

        return self.get_response(request)

    def process_exception(self, request, exception):
        """Auto-create a BugReport and log structured traceback to Loki."""
        try:
            from .models import BugReport

            user = None
            if request.user and getattr(request.user, 'is_authenticated', False):
                user = request.user

            trace_id = get_current_trace_id()

            BugReport.objects.create(
                user=user,
                title=f"{type(exception).__name__}: {str(exception)[:200]}",
                description='Auto-generated from unhandled exception.',
                url=request.build_absolute_uri(),
                trace_id=trace_id,
                user_agent=request.META.get('HTTP_USER_AGENT', '')[:512],
                source='exception',
            )

            frames = [
                {"filename": f.filename, "lineno": f.lineno, "name": f.name, "line": f.line}
                for f in traceback.extract_tb(exception.__traceback__)
            ]
            logger.error(
                "obs.debug.bug_report_middleware_error",
                trace_id=trace_id,
                user=user.username if user else "anonymous",
                method=request.method,
                path=request.path,
                exception_type=type(exception).__name__,
                exception_message=str(exception),
                traceback_frames=frames,
            )
        except Exception:
            logger.debug("obs.debug.bug_report_middleware_error", exc_info=True)

        return None
