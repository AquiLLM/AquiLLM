from __future__ import annotations

from contextlib import nullcontext
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest

from apps.knowledge_graph.projection import outbox


def test_command_dispatch_drains_all_due_pages_with_one_cutoff(monkeypatch):
    batches = iter((2, 2, 1))
    calls = []

    def publish(**kwargs):
        calls.append(kwargs)
        count = next(batches)
        return outbox.OutboxPublishSummaryV1(count, count, 0)

    monkeypatch.setattr(outbox, "publish_projection_outbox", publish)
    assert outbox.dispatch_due_projection_work(page_size=2) == 5
    assert {call["now"] for call in calls} == {calls[0]["now"]}
    assert all(
        call["limit"] == 2 and call["using"] == "projection_state" for call in calls
    )


def test_command_dispatch_stops_on_failed_publication(monkeypatch):
    monkeypatch.setattr(
        outbox,
        "publish_projection_outbox",
        lambda **kwargs: outbox.OutboxPublishSummaryV1(2, 1, 1),
    )
    with pytest.raises(RuntimeError, match="pending"):
        outbox.dispatch_due_projection_work(page_size=2)


class _Row(SimpleNamespace):
    def save(self, **kwargs):
        self.saved.append(kwargs["update_fields"])


def _row():
    return _Row(
        id=uuid4(),
        projection_id=uuid4(),
        operation="project",
        state="pending",
        attempt_count=0,
        next_attempt_at=datetime(2026, 8, 20, tzinfo=UTC),
        published_at=None,
        last_failure_code="",
        saved=[],
    )


def test_broker_failure_remains_durable_and_republishes(monkeypatch):
    row = _row()
    monkeypatch.setattr(outbox, "_atomic", lambda _using: nullcontext())
    monkeypatch.setattr(outbox, "_due_outbox_rows", lambda **_kwargs: (row,))

    def fail(*_args, **_kwargs):
        raise RuntimeError("credential and payload must be redacted")

    monkeypatch.setattr(outbox, "_publish", fail)
    now = datetime(2026, 8, 20, tzinfo=UTC)
    failed = outbox.publish_projection_outbox(limit=10, now=now, using="default")

    assert failed.failed_count == 1
    assert row.state == "pending"
    assert row.last_failure_code == "broker_publish_failed"
    assert "credential" not in repr(failed)

    monkeypatch.setattr(outbox, "_publish", lambda *_args, **_kwargs: None)
    recovered = outbox.publish_projection_outbox(
        limit=10, now=row.next_attempt_at, using="default"
    )

    assert recovered.published_count == 1
    assert row.state == "published" and row.published_at == row.next_attempt_at


def test_outbox_fanout_and_limit_are_bounded_before_query(monkeypatch):
    monkeypatch.setattr(outbox, "_atomic", lambda _using: nullcontext())
    observed = []
    monkeypatch.setattr(
        outbox,
        "_due_outbox_rows",
        lambda **kwargs: observed.append(kwargs["limit"]) or (),
    )

    summary = outbox.publish_projection_outbox(
        limit=5000,
        now=datetime(2026, 8, 20, tzinfo=UTC),
        using="default",
    )

    assert observed == [5000]
    assert summary.attempted_count == 0


def test_outbox_dispatches_exact_project_and_prune_tasks(monkeypatch):
    from apps.knowledge_graph.projection import tasks

    projection_id = uuid4()
    dispatched = []
    monkeypatch.setattr(
        tasks.project_knowledge_graph_projection,
        "delay",
        lambda value: dispatched.append(("project", value)),
    )
    monkeypatch.setattr(
        tasks.prune_knowledge_graph_projection,
        "delay",
        lambda value: dispatched.append(("prune", value)),
    )

    outbox._publish(projection_id, "project")
    outbox._publish(projection_id, "prune")

    assert dispatched == [
        ("project", str(projection_id)),
        ("prune", str(projection_id)),
    ]
