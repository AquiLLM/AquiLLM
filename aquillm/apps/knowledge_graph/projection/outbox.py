from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from django.db import transaction

from apps.knowledge_graph.models import GraphProjectionOutbox

_MAX_LIMIT = 5_000


@dataclass(frozen=True, slots=True)
class OutboxPublishSummaryV1:
    attempted_count: int
    published_count: int
    failed_count: int


def _atomic(using: str):
    return transaction.atomic(using=using)


def _due_outbox_rows(*, limit: int, now: datetime, using: str):
    return tuple(
        GraphProjectionOutbox.objects.using(using)
        .select_for_update(skip_locked=True)
        .filter(state=GraphProjectionOutbox.State.PENDING, next_attempt_at__lte=now)
        .order_by("next_attempt_at", "id")[:limit]
    )


def _publish(projection_id, operation: str) -> None:
    from .tasks import (
        project_knowledge_graph_projection,
        prune_knowledge_graph_projection,
    )

    task = (
        project_knowledge_graph_projection
        if operation == "project"
        else prune_knowledge_graph_projection
    )
    task.delay(str(projection_id))


def publish_projection_outbox(
    *, limit: int, now: datetime, using: str
) -> OutboxPublishSummaryV1:
    if type(limit) is not int or not 1 <= limit <= _MAX_LIMIT:
        raise ValueError("limit must be an integer in 1..5000")
    if type(now) is not datetime or now.tzinfo is not UTC:
        raise ValueError("now must be an exact UTC datetime")
    published = failed = 0
    with _atomic(using):
        rows = _due_outbox_rows(limit=limit, now=now, using=using)
        for row in rows:
            try:
                _publish(row.projection_id, row.operation)
            except Exception:
                row.attempt_count += 1
                row.last_failure_code = "broker_publish_failed"
                row.next_attempt_at = now + timedelta(
                    seconds=min(300, 2 ** min(row.attempt_count, 8))
                )
                row.save(
                    using=using,
                    update_fields=[
                        "attempt_count",
                        "last_failure_code",
                        "next_attempt_at",
                    ],
                )
                failed += 1
            else:
                row.state = GraphProjectionOutbox.State.PUBLISHED
                row.published_at = now
                row.last_failure_code = ""
                row.save(
                    using=using,
                    update_fields=["state", "published_at", "last_failure_code"],
                )
                published += 1
    return OutboxPublishSummaryV1(len(rows), published, failed)
