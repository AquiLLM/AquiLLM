from __future__ import annotations

from contextlib import nullcontext
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

from apps.knowledge_graph.projection import lifecycle


class _Row(SimpleNamespace):
    def save(self, **kwargs):
        self.saved.append(kwargs["update_fields"])


def test_evaluation_activation_completes_without_projection_enqueue(monkeypatch):
    from django.db import transaction

    from apps.knowledge_graph.graph import assembly

    artifact = SimpleNamespace(pk=11, evaluation_only=True)
    monkeypatch.setattr(transaction, "atomic", lambda: nullcontext())
    monkeypatch.setattr(
        assembly,
        "_locked_candidate",
        lambda *_args, **_kwargs: (
            SimpleNamespace(pk=7),
            artifact,
            SimpleNamespace(),
            (artifact,),
        ),
    )
    monkeypatch.setattr(
        assembly,
        "_validate_locked_complete_artifact",
        lambda **_kwargs: "validated",
    )
    monkeypatch.setattr(
        assembly,
        "_swap_active_collection_artifact",
        lambda **_kwargs: None,
    )
    enqueued = []
    monkeypatch.setattr(
        lifecycle,
        "enqueue_collection_projection_locked",
        lambda **kwargs: enqueued.append(kwargs),
    )

    result = assembly.activate_collection_graph(7, 13, "a" * 64)

    assert result == "validated"
    assert enqueued == []


def test_rebuild_resets_published_project_outbox_to_pending(monkeypatch):
    now = datetime(2026, 8, 20, tzinfo=UTC)
    projection = SimpleNamespace(id=uuid4())
    entry = _Row(
        state="published",
        published_at=now,
        next_attempt_at=now,
        last_failure_code="",
        saved=[],
    )

    class Store:
        def using(self, _using):
            return self

        def get_or_create(self, **_kwargs):
            return entry, False

    monkeypatch.setattr(lifecycle.GraphProjectionOutbox, "objects", Store())

    lifecycle._enqueue_outbox(projection, "project", now, "default")

    assert entry.state == "pending"
    assert entry.published_at is None
    assert entry.next_attempt_at == now
    assert entry.saved[-1] == [
        "state",
        "published_at",
        "next_attempt_at",
        "last_failure_code",
    ]
