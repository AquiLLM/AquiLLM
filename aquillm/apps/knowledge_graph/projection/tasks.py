from __future__ import annotations

from time import monotonic
from uuid import UUID, uuid4

from celery import shared_task
from django.conf import settings as django_settings
from django.utils import timezone

from . import maintenance
from . import publication
from .memgraph_driver import MemgraphDriverError
from .outbox import publish_projection_outbox
from .reconciler import (
    prune_graph_projection_generations,
    reconcile_graph_projections,
    reconcile_projection_batch,
)
from .runtime import ProjectionDatabaseAliases, load_projection_runtime_settings
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
    priority=0,
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
    if dry_run is True:
        summary = _run_redacted(
            self,
            lambda: reconcile_graph_projections(
                page_size=size,
                dry_run=True,
                collection_id=collection_id,
            ),
        )
        return {
            "examined_count": summary.examined_count,
            "enqueued_count": summary.enqueued_count,
            "published_count": 0,
        }
    from .reconciler import _collection, _size

    size = _size(size, "page_size")
    audit_size = min(size, maintenance.MAX_ARTIFACTS)
    publication_limit = min(size, maintenance.MAX_OUTBOX_PUBLICATIONS)
    _collection(collection_id)
    if type(dry_run) is not bool:
        raise TypeError("dry_run must be exact")
    result = dict(
        examined_count=0,
        enqueued_count=0,
        published_count=0,
        failure_count=0,
        failure_code=None,
        skipped=False,
    )
    try:
        admission = maintenance.admit(
            maintenance.broker_client(),
            scope="global" if collection_id is None else f"collection:{collection_id}",
            interval=int(
                getattr(django_settings, "KG_MAINTENANCE_INTERVAL_SECONDS", 300)
            ),
        )
    except Exception:
        return {
            **result,
            "failure_code": "maintenance_coordination_unavailable",
            "skipped": True,
        }
    if admission is None:
        return {**result, "skipped": True}
    deadline = monotonic() + maintenance.PASS_SECONDS

    # Failed publications remain durable and due. The next admitted pass retries
    # them; maintenance never produces its own retry/global-audit message flood.
    def flush():
        try:
            summary = publish_projection_outbox(
                limit=publication_limit,
                now=timezone.now(),
                using=ProjectionDatabaseAliases().state,
            )
            result["published_count"] += summary.published_count
            result["failure_count"] += summary.failed_count
        except Exception:
            result["failure_count"] += 1
            result["failure_code"] = "maintenance_outbox_unavailable"

    if monotonic() < deadline:
        flush()
    try:
        if monotonic() < deadline:
            summary = reconcile_projection_batch(
                after_id=admission.cursor,
                page_size=audit_size,
                deadline=deadline,
                collection_id=collection_id,
                save_cursor=admission.save_cursor,
            )
            result["examined_count"] = summary.examined_count
            result["enqueued_count"] = summary.enqueued_count
            result["failure_count"] += summary.failure_count
    except Exception:
        result["failure_count"] += 1
        result["failure_code"] = "maintenance_pass_failed"
    if monotonic() < deadline:
        flush()
    return result


@shared_task(
    bind=True,
    base=publication.ScheduledReconcileTask,
    name=publication.SCHEDULED_TASK,
    queue=_TASK_QUEUE + "-maintenance",
    priority=9,
    serializer="json",
    acks_late=False,
    reject_on_worker_lost=False,
    ignore_result=True,
    max_retries=0,
    soft_time_limit=100,
    time_limit=120,
)
def scheduled_reconcile_knowledge_graph_projections(self, publication_token: str):
    """Token-fenced, early-ack global audit; manual requests use the old task."""
    if publication_token != self.request.id:
        return None
    client = None
    started = False
    try:
        client = publication.broker_client(self.app)
        started = publication.start(client, publication_token, app=self.app)
    except Exception:
        publication.logger.warning("obs.kg.scheduled_reconcile_coordination_unavailable")
    if not started:
        if client is not None:
            client.close()
        return None
    try:
        return reconcile_knowledge_graph_projections.run()
    finally:
        try:
            publication.finish(client, publication_token, app=self.app)
        except Exception:
            publication.logger.warning("obs.kg.scheduled_reconcile_release_unavailable")
        finally:
            client.close()


@shared_task(
    bind=True,
    name="apps.knowledge_graph.projection.tasks.prune_knowledge_graph_projection",
    max_retries=_TASK_MAX_RETRIES,
    queue=_TASK_QUEUE,
    priority=0,
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
    "project_knowledge_graph_projection",
    "prune_knowledge_graph_projection",
    "reconcile_knowledge_graph_projections",
    "scheduled_reconcile_knowledge_graph_projections",
]
