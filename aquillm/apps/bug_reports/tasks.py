import structlog
from celery import shared_task

logger = structlog.stdlib.get_logger(__name__)


@shared_task
def debug_celery_task():
    """DEBUG-only task to verify structlog + OTEL tracing on the Celery worker."""
    logger.info("obs.debug.celery_test_started", status="running")
    logger.warning("obs.debug.celery_test_warning", detail="this is a test warning")
    logger.info("obs.debug.celery_test_finished", status="complete")
    return "ok"
