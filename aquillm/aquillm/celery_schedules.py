"""Pure construction of optional Celery beat schedules."""

from __future__ import annotations


def knowledge_graph_maintenance_schedule(
    *,
    enabled: bool,
    extraction_queue: str,
    projection_queue: str,
    interval_seconds: int,
) -> dict[str, dict[str, object]]:
    if enabled is not True:
        return {}
    return {
        "knowledge-graph-build-recovery": {
            "task": "apps.knowledge_graph.tasks.recover_missing_graph_builds_task",
            "schedule": interval_seconds,
            "options": {"queue": extraction_queue, "priority": 9},
        },
        "knowledge-graph-projection-reconcile": {
            "task": (
                "apps.knowledge_graph.projection.tasks."
                "reconcile_knowledge_graph_projections"
            ),
            "schedule": interval_seconds,
            "options": {"queue": projection_queue},
        },
    }


__all__ = ["knowledge_graph_maintenance_schedule"]
