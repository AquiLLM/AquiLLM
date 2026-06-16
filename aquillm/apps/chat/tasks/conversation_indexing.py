"""Celery task: (re)build ConversationChunk rows for a conversation.

Enqueued from the chat consumer after assistant turns are persisted. A short
countdown plus an updated_at snapshot check debounces rapid successive saves: each
save enqueues a fresh task, older tasks see a stale snapshot and skip, and the
last one runs once the conversation settles. The index service is also
hash-guarded, so redundant runs are cheap no-ops.
"""
from __future__ import annotations

import structlog
from datetime import timezone as dt_timezone
from os import getenv

from celery import shared_task
from django.utils.dateparse import parse_datetime
from django.utils import timezone

logger = structlog.stdlib.get_logger(__name__)


def _index_idle_seconds() -> int:
    try:
        value = int((getenv("CONVERSATION_INDEX_IDLE_SECONDS", "60") or "60").strip())
    except Exception:
        return 60
    return max(5, value)


def _parse_queued_updated_at(value: str | None):
    if not value:
        return None
    parsed = parse_datetime(value)
    if parsed is None:
        return None
    if timezone.is_naive(parsed):
        return timezone.make_aware(parsed, dt_timezone.utc)
    return parsed


def enqueue_index_conversation_task(conversation_id: int, queued_updated_at: str | None) -> None:
    index_conversation_task.apply_async(
        kwargs={
            "conversation_id": conversation_id,
            "queued_updated_at": queued_updated_at,
        },
        countdown=_index_idle_seconds(),
    )


@shared_task(serializer="json")
def index_conversation_task(conversation_id: int, queued_updated_at: str | None = None) -> None:
    from apps.chat.models import WSConversation
    from apps.chat.services.conversation_indexing import index_conversation

    convo = WSConversation.objects.filter(id=conversation_id).first()
    if convo is None:
        return
    expected = _parse_queued_updated_at(queued_updated_at)
    if expected is not None and convo.updated_at != expected:
        # A newer save happened after this task was queued; it enqueued its own
        # task, so let that one do the work.
        return
    index_conversation(conversation_id)


__all__ = ["enqueue_index_conversation_task", "index_conversation_task"]
