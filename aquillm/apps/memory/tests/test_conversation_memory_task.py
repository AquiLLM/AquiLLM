"""Tests for delayed conversation memory scheduling and idle gating."""

from __future__ import annotations

from datetime import timedelta

import pytest
from django.contrib.auth import get_user_model
from django.utils import timezone

from aquillm import tasks as tasks_module
from aquillm.models import WSConversation

User = get_user_model()


@pytest.mark.django_db
def test_enqueue_conversation_memories_task_uses_idle_countdown(monkeypatch):
    captured: list[dict[str, object]] = []

    monkeypatch.setattr(
        tasks_module.create_conversation_memories_task,
        "apply_async",
        lambda **kwargs: captured.append(kwargs),
    )
    monkeypatch.setattr(tasks_module, "_mem0_infer_idle_seconds", lambda: 300, raising=False)

    tasks_module.enqueue_conversation_memories_task(
        conversation_id=14,
        queued_updated_at="2026-03-31T17:00:00+00:00",
    )

    assert captured == [
        {
            "kwargs": {
                "conversation_id": 14,
                "queued_updated_at": "2026-03-31T17:00:00+00:00",
            },
            "countdown": 300,
        }
    ]


@pytest.mark.django_db
def test_create_conversation_memories_task_skips_when_source_conversation_changed(monkeypatch):
    user = User.objects.create_user(username="skipchanged", password="pass")
    convo = WSConversation.objects.create(owner=user, system_prompt="sys")
    original_updated_at = convo.updated_at
    convo.updated_at = original_updated_at + timedelta(seconds=10)
    convo.save(update_fields=["updated_at"])

    called: list[int] = []
    monkeypatch.setattr(
        tasks_module,
        "_run_conversation_memory_creation",
        lambda db_convo: called.append(db_convo.id),
        raising=False,
    )

    tasks_module.create_conversation_memories_task(
        conversation_id=convo.id,
        queued_updated_at=original_updated_at.isoformat(),
    )

    assert called == []


@pytest.mark.django_db
def test_create_conversation_memories_task_reschedules_while_user_is_active_elsewhere(monkeypatch):
    user = User.objects.create_user(username="rescheduleactive", password="pass")
    source = WSConversation.objects.create(owner=user, system_prompt="sys")
    newer = WSConversation.objects.create(owner=user, system_prompt="sys")

    idle_seconds = 300
    now = timezone.now()
    source.updated_at = now - timedelta(minutes=4)
    source.save(update_fields=["updated_at"])
    newer.updated_at = now - timedelta(seconds=30)
    newer.save(update_fields=["updated_at"])

    rescheduled: list[dict[str, object]] = []
    called: list[int] = []

    monkeypatch.setattr(tasks_module, "_mem0_infer_idle_seconds", lambda: idle_seconds, raising=False)
    monkeypatch.setattr(tasks_module, "_utcnow", lambda: now, raising=False)
    monkeypatch.setattr(
        tasks_module,
        "enqueue_conversation_memories_task",
        lambda **kwargs: rescheduled.append(kwargs),
        raising=False,
    )
    monkeypatch.setattr(
        tasks_module,
        "_run_conversation_memory_creation",
        lambda db_convo: called.append(db_convo.id),
        raising=False,
    )

    tasks_module.create_conversation_memories_task(
        conversation_id=source.id,
        queued_updated_at=source.updated_at.isoformat(),
    )

    assert called == []
    assert rescheduled == [
        {
            "conversation_id": source.id,
            "queued_updated_at": source.updated_at.isoformat(),
        }
    ]


@pytest.mark.django_db
def test_create_conversation_memories_task_runs_once_user_is_globally_idle(monkeypatch):
    user = User.objects.create_user(username="globallyidle", password="pass")
    source = WSConversation.objects.create(owner=user, system_prompt="sys")
    newer = WSConversation.objects.create(owner=user, system_prompt="sys")

    idle_seconds = 300
    now = timezone.now()
    source.updated_at = now - timedelta(minutes=12)
    source.save(update_fields=["updated_at"])
    newer.updated_at = now - timedelta(minutes=7)
    newer.save(update_fields=["updated_at"])

    called: list[int] = []

    monkeypatch.setattr(tasks_module, "_mem0_infer_idle_seconds", lambda: idle_seconds, raising=False)
    monkeypatch.setattr(tasks_module, "_utcnow", lambda: now, raising=False)
    monkeypatch.setattr(
        tasks_module,
        "_run_conversation_memory_creation",
        lambda db_convo: called.append(db_convo.id),
        raising=False,
    )

    tasks_module.create_conversation_memories_task(
        conversation_id=source.id,
        queued_updated_at=source.updated_at.isoformat(),
    )

    assert called == [source.id]
