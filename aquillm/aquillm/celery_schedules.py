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


def application_maintenance_schedule(*, enabled: bool) -> dict[str, dict[str, object]]:
    if enabled is not True:
        return {}
    return {
        "document-chunk-publication-recovery": {
            "task": "apps.documents.tasks.chunk_recovery.recover_chunk_publications",
            "schedule": 60,
            "kwargs": {"limit": 25},
            "options": {"queue": "celery"},
        },
        "conversation-memory-recovery": {
            "task": "aquillm.tasks.recover_conversation_memory_jobs",
            "schedule": 60,
            "kwargs": {"limit": 25},
            "options": {"queue": "celery"},
        },
    }


__all__ = ["knowledge_graph_maintenance_schedule", "application_maintenance_schedule"]
