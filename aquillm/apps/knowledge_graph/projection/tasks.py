from __future__ import annotations

from uuid import UUID, uuid4

from celery import shared_task
from django.utils import timezone

from .memgraph_driver import MemgraphDriverError
from .outbox import publish_projection_outbox
from .reconciler import (
    prune_graph_projection_generations,
    reconcile_graph_projections,
)
from .runtime import load_projection_runtime_settings
from .worker import project_generation

_TRANSIENT = (ConnectionError, TimeoutError)
_TASK_SETTINGS = load_projection_runtime_settings()
_TASK_QUEUE = _TASK_SETTINGS.projection_queue
_TASK_MAX_RETRIES = _TASK_SETTINGS.projection_max_attempts - 1


def _run_redacted(task, operation):
    try:
        return operation()
    except Exception as exc:
        transient = isinstance(exc, _TRANSIENT) or (
            isinstance(exc, MemgraphDriverError)
            and exc.code in {"memgraph_read_failed", "memgraph_write_failed"}
        )
        if not transient:
            raise RuntimeError("projection_task_failed") from None
        raise task.retry(
            exc=RuntimeError("projection_task_transient"), countdown=30
        ) from None


def _uuid(value: object) -> UUID:
    if type(value) is not str:
        raise ValueError("projection_id must be a canonical UUID string")
    try:
        parsed = UUID(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("projection_id must be a canonical UUID string") from exc
    if str(parsed) != value:
        raise ValueError("projection_id must be a canonical UUID string")
    return parsed


@shared_task(
    bind=True,
    name="apps.knowledge_graph.projection.tasks.project_knowledge_graph_projection",
    max_retries=_TASK_MAX_RETRIES,
    queue=_TASK_QUEUE,
    acks_late=True,
    reject_on_worker_lost=True,
)
def project_knowledge_graph_projection(self, projection_id: str):
    identifier = _uuid(projection_id)
    outcome = _run_redacted(
        self,
        lambda: project_generation(
            projection_id=identifier,
            lease_owner=f"celery-{self.request.id or uuid4()}",
        ),
    )
    return {"ready": outcome.ready, "failure_code": outcome.failure_code}


@shared_task(
    bind=True,
    name="apps.knowledge_graph.projection.tasks.reconcile_knowledge_graph_projections",
    max_retries=_TASK_MAX_RETRIES,
    queue=_TASK_QUEUE,
    acks_late=True,
)
def reconcile_knowledge_graph_projections(
    self,
    page_size: int | None = None,
    dry_run: bool = False,
    collection_id: int | None = None,
):
    size = _TASK_SETTINGS.projection_batch_size if page_size is None else page_size
    summary = _run_redacted(
        self,
        lambda: reconcile_graph_projections(
            page_size=size,
            dry_run=dry_run,
            collection_id=collection_id,
        ),
    )
    published = (
        None
        if dry_run
        else _run_redacted(
            self,
            lambda: publish_projection_outbox(
                limit=size,
                now=timezone.now(),
                using="default",
            ),
        )
    )
    return {
        "examined_count": summary.examined_count,
        "enqueued_count": summary.enqueued_count,
        "published_count": 0 if published is None else published.published_count,
    }


@shared_task(
    bind=True,
    name="apps.knowledge_graph.projection.tasks.publish_knowledge_graph_projection_outbox",
    max_retries=_TASK_MAX_RETRIES,
    queue=_TASK_QUEUE,
    acks_late=True,
    reject_on_worker_lost=True,
)
def publish_knowledge_graph_projection_outbox(self, limit: int | None = None):
    size = _TASK_SETTINGS.projection_batch_size if limit is None else limit
    summary = _run_redacted(
        self,
        lambda: publish_projection_outbox(
            limit=size,
            now=timezone.now(),
            using="default",
        ),
    )
    return {
        "attempted_count": summary.attempted_count,
        "published_count": summary.published_count,
        "failed_count": summary.failed_count,
    }


@shared_task(
    bind=True,
    name="apps.knowledge_graph.projection.tasks.prune_knowledge_graph_projection",
    max_retries=_TASK_MAX_RETRIES,
    queue=_TASK_QUEUE,
    acks_late=True,
)
def prune_knowledge_graph_projection(
    self,
    projection_id: str | None = None,
    collection_id: int | None = None,
    page_size: int | None = None,
    retain: int | None = None,
    dry_run: bool = False,
):
    identifier = None if projection_id is None else _uuid(projection_id)
    size = _TASK_SETTINGS.projection_batch_size if page_size is None else page_size
    retention = _TASK_SETTINGS.projection_retention if retain is None else retain
    summary = _run_redacted(
        self,
        lambda: prune_graph_projection_generations(
            page_size=size,
            retain=retention,
            dry_run=dry_run,
            projection_id=identifier,
            collection_id=collection_id,
        ),
    )
    return {
        "candidate_count": summary.candidate_count,
        "deleted_count": summary.deleted_count,
    }


__all__ = [
    "publish_knowledge_graph_projection_outbox",
    "project_knowledge_graph_projection",
    "prune_knowledge_graph_projection",
    "reconcile_knowledge_graph_projections",
]
