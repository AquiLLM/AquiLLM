from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest

from apps.knowledge_graph.projection import tasks


def test_projection_task_uses_canonical_uuid_and_returns_redacted_summary(monkeypatch):
    projection_id = uuid4()
    monkeypatch.setattr(
        tasks,
        "project_generation",
        lambda **_kwargs: SimpleNamespace(ready=True, failure_code=None),
    )

    result = tasks.project_knowledge_graph_projection.run(str(projection_id))

    assert result == {"ready": True, "failure_code": None}


def test_projection_task_rejects_noncanonical_payload_without_echoing_it():
    with pytest.raises(ValueError, match="canonical") as captured:
        tasks.project_knowledge_graph_projection.run("SECRET-NOT-A-UUID")
    assert "SECRET" not in repr(captured.value)


def test_reconcile_and_prune_tasks_are_registered_thin_wrappers(monkeypatch):
    monkeypatch.setattr(
        tasks,
        "reconcile_graph_projections",
        lambda **_kwargs: SimpleNamespace(examined_count=2, enqueued_count=1),
    )
    monkeypatch.setattr(
        tasks,
        "prune_graph_projection_generations",
        lambda **_kwargs: SimpleNamespace(candidate_count=3, deleted_count=2),
    )

    reconciled = tasks.reconcile_knowledge_graph_projections.run(10, True)
    pruned = tasks.prune_knowledge_graph_projection.run(None, 10, 2, True)

    assert reconciled == {"examined_count": 2, "enqueued_count": 1}
    assert pruned == {"candidate_count": 3, "deleted_count": 2}
