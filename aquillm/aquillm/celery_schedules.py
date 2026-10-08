"""Pure construction of optional Celery beat schedules."""

from __future__ import annotations


def knowledge_graph_maintenance_schedule(
    *,
    enabled: bool,
    extraction_queue: str,
    projection_queue: str,
    interval_seconds: int,
    pruning_enabled: bool = False,
    pruning_interval_seconds: int = 86400,
) -> dict[str, dict[str, object]]:
    if enabled is not True:
        return {}
    schedule = {
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
    if pruning_enabled is True:
        if (
            type(pruning_interval_seconds) is not int
            or not 86400 <= pruning_interval_seconds <= 604800
        ):
            raise ValueError(
                "pruning_interval_seconds must be an integer from 86400 to 604800"
            )
        schedule["knowledge-graph-artifact-pruning"] = {
            "task": "apps.knowledge_graph.tasks.prune_graph_artifacts_task",
            "schedule": pruning_interval_seconds,
            "options": {"queue": extraction_queue, "priority": 9},
        }
    return schedule


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
