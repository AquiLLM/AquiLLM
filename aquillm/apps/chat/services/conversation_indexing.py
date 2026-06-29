"""Build ConversationChunk rows (turn windows + embeddings) for a conversation.

Modeled on apps.documents.tasks.chunking.create_chunks, but for chat transcripts:
group messages into turn windows (lib.conversations.chunking), batch-embed, and
replace the conversation's chunks transactionally. A transcript hash on
WSConversation guards against re-indexing unchanged conversations.
"""
from __future__ import annotations

import hashlib
import structlog

from django.apps import apps as django_apps
from django.db import transaction

from aquillm.utils import get_embedding, get_embeddings
from apps.chat.models import ConversationChunk, Message, WSConversation
from lib.conversations.chunking import TranscriptMessage, build_turn_windows

logger = structlog.stdlib.get_logger(__name__)

# Roles that carry natural-language content worth indexing. Tool messages hold
# JSON result payloads, not prose, so they're excluded from the transcript.
_INDEXED_ROLES = ("user", "assistant")


def _transcript_messages(conversation: WSConversation) -> list[TranscriptMessage]:
    rows = (
        Message.objects.filter(conversation=conversation, role__in=_INDEXED_ROLES)
        .order_by("sequence_number")
        .values_list("sequence_number", "role", "content")
    )
    return [
        TranscriptMessage(role=role, content=content, sequence_number=seq)
        for seq, role, content in rows
        if (content or "").strip()
    ]


def _transcript_hash(messages: list[TranscriptMessage]) -> str:
    hasher = hashlib.sha256()
    for m in messages:
        hasher.update(f"{m.sequence_number}\x1f{m.role}\x1f{m.content}\x1e".encode("utf-8"))
    return hasher.hexdigest()


def index_conversation(conversation_id: int, *, force: bool = False) -> int:
    """(Re)build ConversationChunk rows for one conversation.

    Returns the number of chunks written (0 if skipped or empty). Idempotent: a
    matching transcript hash short-circuits unless ``force`` is set.
    """
    conversation = WSConversation.objects.filter(id=conversation_id).first()
    if conversation is None:
        logger.warning("index_conversation: no conversation %s", conversation_id)
        return 0

    messages = _transcript_messages(conversation)
    transcript_hash = _transcript_hash(messages)

    if (
        not force
        and conversation.index_complete
        and conversation.indexed_transcript_hash == transcript_hash
    ):
        return ConversationChunk.objects.filter(conversation=conversation).count()

    if not messages:
        ConversationChunk.objects.filter(conversation=conversation).delete()
        WSConversation.objects.filter(pk=conversation.pk).update(
            indexed_transcript_hash=transcript_hash, index_complete=True
        )
        return 0

    app_config = django_apps.get_app_config("aquillm")
    windows = build_turn_windows(
        messages,
        target_size=app_config.chunk_size,
        overlap=app_config.chunk_overlap,
    )
    if not windows:
        ConversationChunk.objects.filter(conversation=conversation).delete()
        WSConversation.objects.filter(pk=conversation.pk).update(
            indexed_transcript_hash=transcript_hash, index_complete=True
        )
        return 0

    texts = [w.content for w in windows]
    embeddings: list = [None] * len(windows)
    try:
        batch = get_embeddings(texts, input_type="search_document")
        if len(batch) != len(windows):
            raise RuntimeError(f"Embedding batch mismatch: expected {len(windows)}, got {len(batch)}")
        embeddings = list(batch)
    except Exception as exc:
        logger.warning("Batch embedding failed for conversation %s: %s", conversation_id, exc)
        for i, text in enumerate(texts):
            try:
                embeddings[i] = get_embedding(text, input_type="search_document")
            except Exception as inner:
                logger.warning(
                    "Per-window embedding failed for conversation %s window %s: %s",
                    conversation_id,
                    i,
                    inner,
                )
                embeddings[i] = None

    chunks = [
        ConversationChunk(
            conversation=conversation,
            content=w.content,
            chunk_number=i,
            start_sequence=w.start_sequence,
            end_sequence=w.end_sequence,
            modality=ConversationChunk.Modality.TEXT,
            metadata=w.metadata,
            embedding=embeddings[i],
        )
        for i, w in enumerate(windows)
    ]

    with transaction.atomic():
        ConversationChunk.objects.filter(conversation=conversation).delete()
        ConversationChunk.objects.bulk_create(chunks)
        WSConversation.objects.filter(pk=conversation.pk).update(
            indexed_transcript_hash=transcript_hash, index_complete=True
        )

    logger.info("Indexed conversation %s into %d chunks", conversation_id, len(chunks))
    return len(chunks)


__all__ = ["index_conversation"]
