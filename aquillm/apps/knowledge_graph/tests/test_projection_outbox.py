from __future__ import annotations

from contextlib import nullcontext
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

from apps.knowledge_graph.projection import outbox


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
