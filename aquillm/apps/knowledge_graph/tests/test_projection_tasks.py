from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest

from apps.knowledge_graph.projection import tasks
from apps.knowledge_graph.projection.memgraph_driver import MemgraphDriverError


def test_projection_task_uses_canonical_uuid_and_returns_redacted_summary(monkeypatch):
    projection_id = uuid4()
    monkeypatch.setattr(
        tasks,
        "project_generation",
        lambda **_kwargs: SimpleNamespace(ready=True, failure_code=None),
    )

    result = tasks.project_knowledge_graph_projection.run(str(projection_id))

    assert result == {"ready": True, "failure_code": None}
    assert (
        tasks.project_knowledge_graph_projection.queue == "knowledge_graph_projection"
    )
    assert tasks.project_knowledge_graph_projection.max_retries == (
        tasks.load_projection_runtime_settings().projection_max_attempts - 1
    )


def test_projection_task_rejects_noncanonical_payload_without_echoing_it():
    with pytest.raises(ValueError, match="canonical") as captured:
        tasks.project_knowledge_graph_projection.run("SECRET-NOT-A-UUID")
    assert "SECRET" not in repr(captured.value)


def test_reconcile_and_prune_tasks_are_registered_thin_wrappers(monkeypatch):
    projection_id = uuid4()
    calls = {}
    monkeypatch.setattr(
        tasks,
        "reconcile_graph_projections",
        lambda **kwargs: (
            calls.update(reconcile=kwargs)
            or SimpleNamespace(examined_count=2, enqueued_count=1)
        ),
    )
    monkeypatch.setattr(
        tasks,
        "prune_graph_projection_generations",
        lambda **kwargs: (
            calls.update(prune=kwargs)
            or SimpleNamespace(candidate_count=3, deleted_count=2)
        ),
    )
    published = []
    monkeypatch.setattr(
        tasks,
        "publish_projection_outbox",
        lambda **kwargs: (
            published.append(kwargs)
            or SimpleNamespace(attempted_count=2, published_count=2, failed_count=0)
        ),
        raising=False,
    )

    reconciled = tasks.reconcile_knowledge_graph_projections.run(10, False, 17)
    pruned = tasks.prune_knowledge_graph_projection.run(
        str(projection_id), None, 10, 2, True
    )

    assert reconciled == {
        "examined_count": 2,
        "enqueued_count": 1,
        "published_count": 2,
    }
    assert pruned == {"candidate_count": 3, "deleted_count": 2}
    assert calls["reconcile"]["collection_id"] == 17
    assert calls["prune"]["projection_id"] == projection_id
    assert calls["prune"]["collection_id"] is None
    assert len(published) == 1


def test_outbox_publisher_task_is_registered_and_redacted(monkeypatch):
    monkeypatch.setattr(
        tasks,
        "publish_projection_outbox",
        lambda **_kwargs: SimpleNamespace(
            attempted_count=3,
            published_count=2,
            failed_count=1,
        ),
        raising=False,
    )

    result = tasks.publish_knowledge_graph_projection_outbox.run(limit=25)

    assert result == {
        "attempted_count": 3,
        "published_count": 2,
        "failed_count": 1,
    }


def test_task_boundary_retries_fixed_redacted_memgraph_failures():
    retried = []
    task = SimpleNamespace(
        retry=lambda **kwargs: retried.append(kwargs) or RuntimeError("retry")
    )

    with pytest.raises(RuntimeError, match="retry"):
        tasks._run_redacted(
            task,
            lambda: (_ for _ in ()).throw(MemgraphDriverError("memgraph_read_failed")),
        )

    assert type(retried[0]["exc"]) is RuntimeError
    assert str(retried[0]["exc"]) == "projection_task_transient"


def test_maintenance_task_defaults_come_from_frozen_projection_config(monkeypatch):
    observed = {}
    monkeypatch.setattr(
        tasks,
        "_TASK_SETTINGS",
        SimpleNamespace(projection_batch_size=17, projection_retention=9),
    )
    monkeypatch.setattr(
        tasks,
        "prune_graph_projection_generations",
        lambda **kwargs: (
            observed.update(kwargs)
            or SimpleNamespace(candidate_count=0, deleted_count=0)
        ),
    )

    tasks.prune_knowledge_graph_projection.run()

    assert observed["page_size"] == 17
    assert observed["retain"] == 9
