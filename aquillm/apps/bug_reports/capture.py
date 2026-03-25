"""Context manager for logging handled exceptions to the bug report system.

Usage (sync):
    with capture_exception(user=request.user):
        do_something_risky()

Usage (async):
    async with capture_exception(user=self.user):
        await do_something_risky()

The exception is re-raised after being recorded so that the caller's own
error handling still runs normally.
"""
import logging
import traceback
from contextlib import contextmanager, asynccontextmanager

from .activity import get_activity_log, now_iso

logger = logging.getLogger(__name__)


def _create_report(exception, user=None, context=''):
    """Create a BugReport + StackTrace from a caught exception."""
    try:
        from .models import BugReport, StackTrace

        activity = []
        if user and getattr(user, 'is_authenticated', False):
            activity = get_activity_log(user.id)

        title = f"{type(exception).__name__}: {str(exception)[:200]}"
        if context:
            title = f"[{context}] {title}"

        report = BugReport.objects.create(
            user=user if user and getattr(user, 'is_authenticated', False) else None,
            title=title,
            description='Auto-generated from handled exception.',
            activity_log=activity,
            source='exception',
        )
        StackTrace.objects.create(
            bug_report=report,
            exception_type=type(exception).__name__,
            exception_message=str(exception),
            traceback_text=''.join(traceback.format_exception(exception)),
            request_method='',
            request_path=context,
        )
    except Exception:
        logger.debug("Failed to create bug report from handled exception", exc_info=True)


@contextmanager
def capture_exception(user=None, context=''):
    """Sync context manager that logs exceptions then re-raises them.

    Args:
        user: The Django User associated with the operation (optional).
        context: A short label for where this happened, e.g. 'chat.receive'.
    """
    try:
        yield
    except Exception as exc:
        _create_report(exc, user=user, context=context)
        raise


@asynccontextmanager
async def async_capture_exception(user=None, context=''):
    """Async context manager that logs exceptions then re-raises them.

    Args:
        user: The Django User associated with the operation (optional).
        context: A short label for where this happened, e.g. 'chat.receive'.
    """
    try:
        yield
    except Exception as exc:
        from channels.db import database_sync_to_async
        await database_sync_to_async(_create_report)(exc, user=user, context=context)
        raise
