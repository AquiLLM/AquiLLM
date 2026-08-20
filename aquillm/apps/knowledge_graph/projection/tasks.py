from __future__ import annotations

from uuid import UUID, uuid4

from celery import shared_task

from .reconciler import (
    project_generation,
    prune_graph_projection_generations,
    reconcile_graph_projections,
)

_TRANSIENT = (ConnectionError, TimeoutError)


def _run_redacted(task, operation):
    try:
        return operation()
    except _TRANSIENT:
        raise task.retry(
            exc=RuntimeError("projection_task_transient"), countdown=30
        ) from None
    except Exception:
        raise RuntimeError("projection_task_failed") from None


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
    max_retries=3,
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
    max_retries=3,
    acks_late=True,
)
def reconcile_knowledge_graph_projections(
    self, page_size: int = 500, dry_run: bool = False
):
    summary = _run_redacted(
        self,
        lambda: reconcile_graph_projections(page_size=page_size, dry_run=dry_run),
    )
    return {
        "examined_count": summary.examined_count,
        "enqueued_count": summary.enqueued_count,
    }


@shared_task(
    bind=True,
    name="apps.knowledge_graph.projection.tasks.prune_knowledge_graph_projection",
    max_retries=3,
    acks_late=True,
)
def prune_knowledge_graph_projection(
    self,
    projection_id: str | None = None,
    page_size: int = 500,
    retain: int = 2,
    dry_run: bool = False,
):
    if projection_id is not None:
        _uuid(projection_id)
    summary = _run_redacted(
        self,
        lambda: prune_graph_projection_generations(
            page_size=page_size, retain=retain, dry_run=dry_run
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
]
