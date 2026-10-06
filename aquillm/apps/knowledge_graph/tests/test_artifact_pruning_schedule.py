import pytest

from aquillm.celery_schedules import knowledge_graph_maintenance_schedule


def schedule(**overrides):
    options = dict(
        enabled=True,
        extraction_queue="exact-extraction",
        projection_queue="exact-projection",
        interval_seconds=300,
    )
    options.update(overrides)
    return knowledge_graph_maintenance_schedule(**options)


def test_pruning_defaults_off_and_preserves_existing_entries():
    assert schedule(pruning_enabled=False) == schedule()
    assert len(schedule()) == 2


def test_pruning_requires_both_scheduler_gates():
    assert schedule(enabled=False, pruning_enabled=True) == {}
    enabled = schedule(pruning_enabled=True)
    pruning = enabled.pop("knowledge-graph-artifact-pruning")
    assert enabled == schedule()
    assert pruning == {
        "task": "apps.knowledge_graph.tasks.prune_graph_artifacts_task",
        "schedule": 86400,
        "options": {"queue": "exact-extraction", "priority": 9},
    }


@pytest.mark.parametrize("interval", [86400, 172800, 604800])
def test_pruning_accepts_daily_to_weekly_interval(interval):
    assert schedule(pruning_enabled=True, pruning_interval_seconds=interval)[
        "knowledge-graph-artifact-pruning"
    ]["schedule"] == interval


@pytest.mark.parametrize("interval", [True, False, 86400.0, "86400", None, 86399, 604801])
def test_pruning_rejects_invalid_interval(interval):
    with pytest.raises(ValueError, match="pruning_interval_seconds"):
        schedule(pruning_enabled=True, pruning_interval_seconds=interval)
