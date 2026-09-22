from __future__ import annotations

import importlib
import sys
from unittest.mock import Mock

from django.conf import settings
from test_tasks import _tasks_module

DOCUMENT_ID = "12345678-1234-4234-9234-123456789abc"
SOURCE_HASH = "a" * 64
DOCUMENT_BUILD_KEY = "b" * 64
COLLECTION_ID = 17
AGGREGATE_SOURCE_SIGNATURE = "c" * 64
COLLECTION_BUILD_KEY = "d" * 64
REQUEST_ID = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"

def test_pruning_task_is_low_priority_lazy_and_unscheduled() -> None:
    sys.modules.pop("apps.knowledge_graph.services.pruning", None)
    tasks = _tasks_module()

    assert tasks.prune_graph_artifacts_task.queue == settings.KG_EXTRACTION_QUEUE
    assert tasks.prune_graph_artifacts_task.priority == 9
    assert tasks.prune_graph_artifacts_task.acks_late is True
    assert tasks.prune_graph_artifacts_task.reject_on_worker_lost is True
    assert tasks.prune_graph_artifacts_task.ignore_result is True
    assert tasks.prune_graph_artifacts_task.serializer == "json"
    assert "apps.knowledge_graph.services.pruning" not in sys.modules
    schedules = getattr(settings, "CELERY_BEAT_SCHEDULE", {})
    assert all(
        entry.get("task") != tasks.prune_graph_artifacts_task.name
        for entry in schedules.values()
    )


def test_pruning_runs_independently_when_graph_builds_are_disabled(
    monkeypatch,
) -> None:
    tasks = _tasks_module()
    monkeypatch.setenv("KG_BUILD_ENABLED", "0")
    pruning = importlib.import_module("apps.knowledge_graph.services.pruning")

    calls = []
    monkeypatch.setattr(
        pruning,
        "prune_graph_artifacts",
        lambda **kwargs: calls.append(kwargs) or {"artifact_count": 0},
    )

    result = tasks.prune_graph_artifacts_task.run()

    assert result == {"artifact_count": 0}
    assert calls == [{"execute": True}]


def test_graph_recovery_task_is_low_priority_and_uses_exact_extraction_queue():
    tasks = _tasks_module()

    assert tasks.recover_missing_graph_builds_task.queue == settings.KG_EXTRACTION_QUEUE
    assert tasks.recover_missing_graph_builds_task.priority == 9
    assert tasks.recover_missing_graph_builds_task.acks_late is True
    assert tasks.recover_missing_graph_builds_task.reject_on_worker_lost is True
    assert tasks.recover_missing_graph_builds_task.ignore_result is True
    assert tasks.recover_missing_graph_builds_task.serializer == "json"


def test_graph_recovery_task_publishes_one_bounded_continuation(monkeypatch):
    tasks = _tasks_module()
    recovery = importlib.import_module("apps.knowledge_graph.graph.recovery")
    monkeypatch.setenv("KG_BUILD_ENABLED", "1")
    monkeypatch.setattr(settings, "KG_MAINTENANCE_SCHEDULER_ENABLED", True)
    monkeypatch.setattr(settings, "KG_GRAPH_RECOVERY_PAGE_SIZE", 25)
    next_cursor = {
        "phase": "collections",
        "document_model_index": 0,
        "last_pk": 17,
    }
    monkeypatch.setattr(
        recovery,
        "recover_graph_builds_page",
        lambda cursor, page_size: {
            "examined_count": 1,
            "current_count": 0,
            "published_count": 1,
            "dependency_pending_count": 0,
            "publish_failed_count": 0,
            "next_cursor": next_cursor,
        },
    )
    apply_async = Mock()
    monkeypatch.setattr(
        tasks.recover_missing_graph_builds_task, "apply_async", apply_async
    )

    result = tasks.recover_missing_graph_builds_task.run(cursor=None, page_size=None)

    assert result["next_cursor"] == next_cursor
    apply_async.assert_called_once_with(
        kwargs={"cursor": next_cursor, "page_size": 25},
        countdown=1,
        queue=settings.KG_EXTRACTION_QUEUE,
        priority=9,
    )


def test_graph_recovery_task_is_disabled_without_maintenance_gate(monkeypatch):
    tasks = _tasks_module()
    recovery = importlib.import_module("apps.knowledge_graph.graph.recovery")
    monkeypatch.setenv("KG_BUILD_ENABLED", "1")
    monkeypatch.setattr(settings, "KG_MAINTENANCE_SCHEDULER_ENABLED", False)
    run = Mock(side_effect=AssertionError("disabled recovery ran"))
    monkeypatch.setattr(recovery, "recover_graph_builds_page", run)

    assert tasks.recover_missing_graph_builds_task.run() is None
    run.assert_not_called()


def test_enabled_maintenance_beat_uses_exact_worker_queues():
    from aquillm.celery_schedules import knowledge_graph_maintenance_schedule

    schedule = knowledge_graph_maintenance_schedule(
        enabled=True,
        extraction_queue="exact-extraction",
        projection_queue="exact-projection",
        interval_seconds=300,
    )

    assert schedule == {
        "knowledge-graph-build-recovery": {
            "task": "apps.knowledge_graph.tasks.recover_missing_graph_builds_task",
            "schedule": 300,
            "options": {"queue": "exact-extraction", "priority": 9},
        },
        "knowledge-graph-projection-reconcile": {
            "task": (
                "apps.knowledge_graph.projection.tasks."
                "reconcile_knowledge_graph_projections"
            ),
            "schedule": 300,
            "options": {"queue": "exact-projection"},
        },
    }
    assert knowledge_graph_maintenance_schedule(
        enabled=False,
        extraction_queue="exact-extraction",
        projection_queue="exact-projection",
        interval_seconds=300,
    ) == {}
