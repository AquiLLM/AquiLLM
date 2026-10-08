"""Tests for delayed conversation memory scheduling and idle gating."""

from __future__ import annotations

from datetime import timedelta

import pytest
from django.contrib.auth import get_user_model
from django.utils import timezone

from aquillm import tasks as tasks_module
from aquillm.models import WSConversation
from apps.memory.models import ConversationMemoryJob

User = get_user_model()


@pytest.mark.django_db(transaction=True)
def test_enqueue_conversation_memories_task_uses_idle_countdown(monkeypatch):
    captured: list[dict[str, object]] = []
    user = User.objects.create_user(username="enqueue-memory")
    convo = WSConversation.objects.create(owner=user, system_prompt="sys")

    monkeypatch.setattr(
        tasks_module.create_conversation_memories_task,
        "apply_async",
        lambda **kwargs: captured.append(kwargs),
    )
    monkeypatch.setattr(tasks_module, "_mem0_infer_idle_seconds", lambda: 300, raising=False)

    tasks_module.enqueue_conversation_memories_task(
        conversation_id=convo.id,
        queued_updated_at="2026-03-31T17:00:00+00:00",
    )

    assert captured == [
        {
            "kwargs": {
                "conversation_id": convo.id,
                "queued_job_token": str(ConversationMemoryJob.objects.get(conversation=convo).token),
            },
            "countdown": 300,
        }
    ]


@pytest.mark.django_db(transaction=True)
def test_create_conversation_memories_task_defers_until_metadata_activity_is_idle(monkeypatch):
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
    job = ConversationMemoryJob.objects.get(conversation=convo)
    assert job.state == 'pending'
    assert job.next_attempt_at > timezone.now()


@pytest.mark.django_db(transaction=True)
def test_create_conversation_memories_task_reschedules_while_user_is_active_elsewhere(monkeypatch):
    user = User.objects.create_user(username="rescheduleactive", password="pass")
    source = WSConversation.objects.create(owner=user, system_prompt="sys")
    newer = WSConversation.objects.create(owner=user, system_prompt="sys")

    idle_seconds = 300
    now = timezone.now()
    source.updated_at = now - timedelta(minutes=4)
    WSConversation.objects.filter(pk=source.pk).update(updated_at=source.updated_at)
    newer.updated_at = now - timedelta(seconds=30)
    WSConversation.objects.filter(pk=newer.pk).update(updated_at=newer.updated_at)

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

    assert called == []
    job = ConversationMemoryJob.objects.get(conversation=source)
    assert job.state == 'pending'
    assert job.desired_transcript_hash == tasks_module._memory_transcript_hash(source)
    assert job.next_attempt_at >= now + timedelta(seconds=idle_seconds)


@pytest.mark.django_db(transaction=True)
def test_create_conversation_memories_task_runs_once_user_is_globally_idle(monkeypatch):
    user = User.objects.create_user(username="globallyidle", password="pass")
    source = WSConversation.objects.create(owner=user, system_prompt="sys")
    newer = WSConversation.objects.create(owner=user, system_prompt="sys")

    idle_seconds = 300
    now = timezone.now()
    source.updated_at = now - timedelta(minutes=12)
    WSConversation.objects.filter(pk=source.pk).update(updated_at=source.updated_at)
    newer.updated_at = now - timedelta(minutes=7)
    WSConversation.objects.filter(pk=newer.pk).update(updated_at=newer.updated_at)

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
