from __future__ import annotations

from datetime import timedelta
import hashlib
import json
from os import getenv

from celery import shared_task
from django.utils import timezone

from apps.documents.tasks.embedding_index import build_embedding_generation
from apps.ingestion.tasks import publish_ingestion_output, recover_ingestion_outputs


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
    user_content: str | None = None,
    assistant_content: str | None = None,
    *, source_conversation_id=None, source_message_uuid=None, source_content_digest=None,
    provider_memory_ids=(), provider_wrote=False, provider_receipt_complete=False,
) -> None:
    from .memory import promote_profile_facts_for_turn

    from apps.chat.models import Message
    from apps.memory.services.profile_facts import source_is_current
    if not source_is_current(user_id=user_id, conversation_id=source_conversation_id,
                            message_uuid=source_message_uuid, digest=source_content_digest):
        # Legacy raw-string jobs cannot mint attribution. A late provider receipt
        # is handled at the synchronous source-bound write boundary instead.
        return
    source = Message.objects.get(conversation_id=source_conversation_id, message_uuid=source_message_uuid, role='user')
    promote_profile_facts_for_turn(
        user_id=user_id,
        user_content=source.content, assistant_content='',
        source_conversation_id=source_conversation_id, source_message_uuid=source_message_uuid,
        source_content_digest=source_content_digest, provider_memory_ids=provider_memory_ids, provider_wrote=provider_wrote, provider_receipt_complete=provider_receipt_complete,
    )


@shared_task(serializer='json', queue='memory-promotion')
def reconcile_profile_memory_task(fact_id: int) -> None:
    from apps.memory.jobs import execute_profile_reconciliation
    execute_profile_reconciliation(fact_id)


@shared_task(serializer='json', ignore_result=True)
def recover_profile_memory_reconciliation_jobs(limit=25):
    from apps.memory.jobs import recover_profile_reconciliations
    recover_profile_reconciliations(limit)


@shared_task(serializer="json")
def ingest_uploaded_file_task(item_id: int) -> None:
    from .task_ingest_uploaded import run_ingest_uploaded_file

    run_ingest_uploaded_file(item_id)
