from __future__ import annotations

from datetime import timedelta
import hashlib
import json
from os import getenv

from celery import shared_task
from django.utils import timezone


def _mem0_infer_idle_seconds() -> int:
    try:
        value = int((getenv("MEM0_INFER_IDLE_SECONDS", "300") or "300").strip())
    except Exception:
        return 300
    return max(30, value)


def _utcnow():
    return timezone.now()


def _memory_transcript_hash(convo) -> str:
    rows = list(convo.db_messages.order_by("sequence_number").values_list(
        "sequence_number", "role", "content", "message_uuid"
    ))
    return hashlib.sha256(json.dumps(rows, default=str).encode()).hexdigest()


def _run_conversation_memory_creation(convo) -> None:
    from .memory import create_episodic_memories_for_conversation

    create_episodic_memories_for_conversation(convo)


def enqueue_conversation_memories_task(
    conversation_id: int,
    queued_updated_at: str | None = None,
) -> None:
    from apps.memory.jobs import request_memory_job
    request_memory_job(conversation_id)


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
    queued_transcript_hash: str | None = None,
    queued_job_token: str | None = None,
) -> None:
    from apps.memory.jobs import execute_memory_job
    execute_memory_job(conversation_id, queued_job_token)


@shared_task(serializer="json", ignore_result=True)
def recover_conversation_memory_jobs(limit=25):
    from apps.memory.jobs import recover_memory_jobs
    return recover_memory_jobs(limit=limit)


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
