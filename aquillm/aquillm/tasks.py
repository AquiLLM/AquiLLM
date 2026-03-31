from __future__ import annotations

from datetime import timedelta, timezone as dt_timezone
from os import getenv

from celery import shared_task
from django.utils import timezone
from django.utils.dateparse import parse_datetime


def _mem0_infer_idle_seconds() -> int:
    try:
        value = int((getenv("MEM0_INFER_IDLE_SECONDS", "300") or "300").strip())
    except Exception:
        return 300
    return max(30, value)


def _utcnow():
    return timezone.now()


def _parse_queued_updated_at(value: str | None):
    if not value:
        return None
    parsed = parse_datetime(value)
    if parsed is None:
        return None
    if timezone.is_naive(parsed):
        return timezone.make_aware(parsed, dt_timezone.utc)
    return parsed


def _run_conversation_memory_creation(convo) -> None:
    from .memory import create_episodic_memories_for_conversation

    create_episodic_memories_for_conversation(convo)


def enqueue_conversation_memories_task(
    conversation_id: int,
    queued_updated_at: str | None,
) -> None:
    create_conversation_memories_task.apply_async(
        kwargs={
            "conversation_id": conversation_id,
            "queued_updated_at": queued_updated_at,
        },
        countdown=_mem0_infer_idle_seconds(),
    )


def _conversation_still_matches_snapshot(convo, queued_updated_at: str | None) -> bool:
    expected = _parse_queued_updated_at(queued_updated_at)
    if expected is None:
        return True
    return convo.updated_at == expected


def _user_is_globally_idle(user_id: int, *, now, idle_seconds: int) -> bool:
    from .models import WSConversation

    latest_activity = (
        WSConversation.objects.filter(owner_id=user_id)
        .order_by("-updated_at")
        .values_list("updated_at", flat=True)
        .first()
    )
    if latest_activity is None:
        return True
    return latest_activity <= now - timedelta(seconds=idle_seconds)


@shared_task(serializer="json")
def create_conversation_memories_task(
    conversation_id: int,
    queued_updated_at: str | None = None,
) -> None:
    from .models import WSConversation

    convo = WSConversation.objects.filter(id=conversation_id).first()
    if convo is None:
        return
    if not _conversation_still_matches_snapshot(convo, queued_updated_at):
        return

    now = _utcnow()
    idle_seconds = _mem0_infer_idle_seconds()
    if not _user_is_globally_idle(convo.owner_id, now=now, idle_seconds=idle_seconds):
        enqueue_conversation_memories_task(
            conversation_id=conversation_id,
            queued_updated_at=convo.updated_at.isoformat(),
        )
        return

    _run_conversation_memory_creation(convo)


@shared_task(serializer="json", queue="memory-promotion")
def promote_profile_facts_task(
    user_id: int,
    user_content: str,
    assistant_content: str,
) -> None:
    from .memory import promote_profile_facts_for_turn

    promote_profile_facts_for_turn(
        user_id=user_id,
        user_content=user_content,
        assistant_content=assistant_content,
    )


@shared_task(serializer="json")
def ingest_uploaded_file_task(item_id: int) -> None:
    from .task_ingest_uploaded import run_ingest_uploaded_file

    run_ingest_uploaded_file(item_id)
