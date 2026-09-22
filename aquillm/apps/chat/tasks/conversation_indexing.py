"""Celery task: (re)build ConversationChunk rows for a conversation.

Enqueued after assistant turns are persisted. A delayed transcript hash snapshot
debounces message changes while allowing metadata-only saves. Stale snapshots
queue a replacement instead of assuming another save scheduled one. The index
service is also hash-guarded, so redundant runs are cheap no-ops.
"""
from __future__ import annotations

from os import getenv

import structlog
from celery import shared_task

logger = structlog.stdlib.get_logger(__name__)


def _index_idle_seconds() -> int:
    try:
        value = int((getenv("CONVERSATION_INDEX_IDLE_SECONDS", "60") or "60").strip())
    except Exception:
        return 60
    return max(5, value)


def enqueue_index_conversation_task(
    conversation_id: int, queued_updated_at: str | None = None
) -> None:
    from apps.chat.models import WSConversation
    from apps.chat.services.conversation_indexing import conversation_transcript_hash

    convo = WSConversation.objects.filter(id=conversation_id).first()
    if convo is None:
        return
    index_conversation_task.apply_async(
        kwargs={
            "conversation_id": conversation_id,
            "queued_transcript_hash": conversation_transcript_hash(convo),
        },
        countdown=_index_idle_seconds(),
    )


@shared_task(serializer="json")
def index_conversation_task(
    conversation_id: int,
    queued_updated_at: str | None = None,
    queued_transcript_hash: str | None = None,
) -> None:
    from apps.chat.models import WSConversation
    from apps.chat.services.conversation_indexing import (
        conversation_transcript_hash,
        index_conversation,
    )

    convo = WSConversation.objects.filter(id=conversation_id).first()
    if convo is None:
        return
    current_hash = conversation_transcript_hash(convo)
    if queued_transcript_hash is not None and queued_transcript_hash != current_hash:
        if not convo.index_complete or convo.indexed_transcript_hash != current_hash:
            enqueue_index_conversation_task(conversation_id)
        return
    # Legacy jobs carry only updated_at. Metadata changes must not cancel them.
    index_conversation(conversation_id)
    convo = WSConversation.objects.filter(id=conversation_id).first()
    if convo is not None and (
        not convo.index_complete
        or convo.indexed_transcript_hash != conversation_transcript_hash(convo)
    ):
        # A transcript may have changed while embeddings were being generated.
        enqueue_index_conversation_task(conversation_id)


__all__ = ["enqueue_index_conversation_task", "index_conversation_task"]
