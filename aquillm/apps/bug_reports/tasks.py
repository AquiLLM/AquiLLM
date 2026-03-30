import structlog
from celery import shared_task

logger = structlog.stdlib.get_logger(__name__)


@shared_task
def debug_celery_task():
    """DEBUG-only task to verify structlog + OTEL tracing on the Celery worker."""
    logger.info("debug_celery_task_started", status="running")
    logger.warning("debug_celery_task_warning", detail="this is a test warning")
    logger.info("debug_celery_task_finished", status="complete")
    return "ok"
