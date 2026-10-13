"""
User memory: profile facts (stable preferences) + episodic semantic memory (retrieved past exchanges).

- UserMemoryFact: injected into system on every chat for this user.
- EpisodicMemory: embedded past turns; we retrieve top-k by similarity to current message and inject.

Optional backend:
- MEM0 ("MEMORY_BACKEND=mem0"): use Mem0 for episodic retrieval/write with local fallback.

This module integrates lib/memory (pure Python) with Django models.
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING, Optional

import structlog
from channels.db import database_sync_to_async
from django.contrib.auth.models import User
from django.db import IntegrityError
from pgvector.django import L2Distance

# Import from lib/memory for pure Python operations
from lib.memory import (
    EPISODIC_TOP_K,
    MEM0_DUAL_WRITE_LOCAL,
    RetrievedEpisodicMemory,
    add_mem0_memory_with_client,
    clean_stable_facts,
    extract_stable_facts,
    format_memories_for_system,
    has_remember_intent,
    heuristic_facts_from_turn,
    normalize_remember_fact,
    parse_memory_intent, validate_fact_candidates,
    search_mem0_episodic_memories,
    search_mem0_episodic_memories_async,
    use_mem0,
)

from .memory_categories import (
    _categorize_profile_fact as _categorize_profile_fact,
)
from .models import EpisodicMemory, UserMemoryFact
from .utils import get_embedding

if TYPE_CHECKING:
    from .llm import Conversation

logger = structlog.stdlib.get_logger(__name__)

DEFAULT_MEMORY_RETRIEVAL_TIMEOUT_SECONDS = 2.0

# Re-export for backward compatibility
__all__ = [
    'RetrievedEpisodicMemory',
    'get_user_profile_facts',
    'get_episodic_memories',
    'get_episodic_memories_async',
    'format_memories_for_system',
    'get_last_user_message_text',
    'augment_conversation_with_memory',
    'augment_conversation_with_memory_async',
    'create_episodic_memories_for_conversation',
]


def _is_duplicate_episodic_memory_error(exc: IntegrityError) -> bool:
    """Check whether an IntegrityError is the assistant-message dedupe race."""
    message = str(exc)
    return "unique_episodic_per_assistant_msg" in message


def promote_profile_facts_for_turn(
    user_id: int,
    user_content: str,
    assistant_content: str,
    *, source_conversation_id=None, source_message_uuid=None, source_content_digest=None,
    provider_memory_ids=(), provider_wrote=False, provider_receipt_complete=False,
) -> int:
    """Extract and persist durable facts for a completed turn."""
    from apps.memory.services.profile_facts import source_is_current, promote_source_payload
    intent = parse_memory_intent(user_content)
    if intent.kind != 'affirmative' or source_conversation_id is None or source_message_uuid is None or not source_content_digest:
        return 0
    user = User.objects.filter(id=user_id).first()
    if user is None:
        return 0
    current = source_is_current(user_id=user_id, conversation_id=source_conversation_id,
                                message_uuid=source_message_uuid, digest=source_content_digest)
    if current:
        # Candidates are constrained to source spans; durable text is always the
        # accepted source payload. Neither assistant text nor a model receipt
        # establishes provenance. Empty extraction has only this scoped fallback.
        candidates = validate_fact_candidates(user_content, extract_stable_facts(user_content, ''))
        logger.info('obs.memory.profile_candidates_validated', candidate_count=len(candidates), scoped_fallback=not bool(candidates))
    return promote_source_payload(user=user, user_content=user_content,
        source_conversation_id=source_conversation_id, source_message_uuid=source_message_uuid,
        source_content_digest=source_content_digest, provider_memory_ids=provider_memory_ids, provider_wrote=provider_wrote, provider_receipt_complete=provider_receipt_complete)


def _enqueue_profile_fact_promotion(
    user_id: int,
    user_content: str,
    assistant_content: str,
    *, source_conversation_id=None, source_message_uuid=None, source_content_digest=None,
    provider_memory_ids=(), provider_wrote=False, provider_receipt_complete=False,
) -> None:
    """Send durable-fact promotion to a lower-priority worker queue."""
    try:
        from .tasks import promote_profile_facts_task

        if parse_memory_intent(user_content).kind != 'affirmative':
            return
        promote_profile_facts_task.delay(user_id=user_id, source_conversation_id=source_conversation_id,
            source_message_uuid=source_message_uuid, source_content_digest=source_content_digest,
            provider_memory_ids=list(provider_memory_ids), provider_wrote=provider_wrote, provider_receipt_complete=provider_receipt_complete)
    except Exception as exc:
        logger.warning(
            "obs.memory.promotion_enqueue_error",
            error_type=type(exc).__name__,
        )
        fact_count = promote_profile_facts_for_turn(
            user_id=user_id,
            user_content=user_content,
            assistant_content=assistant_content,
            source_conversation_id=source_conversation_id, source_message_uuid=source_message_uuid,
            source_content_digest=source_content_digest, provider_memory_ids=provider_memory_ids, provider_wrote=provider_wrote, provider_receipt_complete=provider_receipt_complete,
        )
        logger.info(
            "obs.memory.inline_promotion_completed",
            fact_count=fact_count,
        )


def _add_mem0_memory(
    user: User,
    user_content: str,
    assistant_content: str,
    conversation_id: int,
    assistant_message_uuid: str,
    *, source_message_uuid=None, source_content_digest=None,
) -> None:
    """Source-bound write with a bounded, complete provider receipt."""
    from . import memory as api
    from .memory_write import add_source_memory
    add_source_memory(api, user, user_content, assistant_content, conversation_id, assistant_message_uuid,
                      source_message_uuid=source_message_uuid, source_content_digest=source_content_digest)


def get_user_profile_facts(user: User):
    """Return all profile facts for the user (tone, goals, project, etc.)."""
    from apps.memory.services.profile_facts import get_active_profile_facts
    return get_active_profile_facts(user)


def _get_episodic_memories_pgvector(
    user: User,
    query: str,
    top_k: int,
    exclude_conversation_id: Optional[int],
):
    """Local pgvector episodic retrieval (no Mem0)."""
    qs = EpisodicMemory.objects.filter(user=user).exclude(embedding__isnull=True)
    if exclude_conversation_id is not None:
        qs = qs.exclude(conversation_id=exclude_conversation_id)
    try:
        embedding = get_embedding(query, input_type='search_query')
        return list(qs.order_by(L2Distance('embedding', embedding))[:top_k])
    except Exception:
        return []


def get_episodic_memories(
    user: User,
    query: str,
    top_k: int = EPISODIC_TOP_K,
    exclude_conversation_id: Optional[int] = None,
):
    """
    Retrieve top-k episodic memories for this user by similarity to `query`.
    Excludes memories from the current conversation to avoid injecting the same thread.
    """
    if not query or not query.strip():
        return []
    if use_mem0():
        mem0_results = search_mem0_episodic_memories(
            user_id=str(user.id),
            query=query.strip(),
            top_k=top_k,
            exclude_conversation_id=exclude_conversation_id,
        )
        if mem0_results:
            from apps.memory.services.profile_facts import filter_reconciled_memories
            return filter_reconciled_memories(user, mem0_results)
    from apps.memory.services.profile_facts import filter_reconciled_memories
    return filter_reconciled_memories(user, _get_episodic_memories_pgvector(
        user, query.strip(), top_k, exclude_conversation_id
    ))


async def get_episodic_memories_async(
    user: User,
    query: str,
    top_k: int = EPISODIC_TOP_K,
    exclude_conversation_id: Optional[int] = None,
):
    """Async episodic retrieval: Mem0 async SDK first, then pgvector via sync_to_async."""
    if not query or not query.strip():
        return []
    if use_mem0():
        mem0_results = await search_mem0_episodic_memories_async(
            user_id=str(user.id),
            query=query.strip(),
            top_k=top_k,
            exclude_conversation_id=exclude_conversation_id,
        )
        if mem0_results:
            from apps.memory.services.profile_facts import filter_reconciled_memories
            return await database_sync_to_async(filter_reconciled_memories)(user, mem0_results)
    # Embedding is a blocking network call followed by ORM work. Keep it off
    # Channels' thread-sensitive database lane so a latency-budget cancellation
    # is not forced to wait for the underlying embedding request to return.
    memories = await database_sync_to_async(
        _get_episodic_memories_pgvector,
        thread_sensitive=False,
    )(
        user, query.strip(), top_k, exclude_conversation_id
    )
    from apps.memory.services.profile_facts import filter_reconciled_memories
    return await database_sync_to_async(filter_reconciled_memories)(user, memories)


def get_last_user_message_text(convo: 'Conversation') -> str:
    """Get the content of the most recent user message for use as retrieval query."""
    from .llm import UserMessage

    for msg in reversed(convo.messages):
        if isinstance(msg, UserMessage):
            return msg.content or ""
    return ""


def _memory_retrieval_timeout_seconds() -> float:
    raw_value = os.getenv(
        "MEMORY_RETRIEVAL_TIMEOUT_SECONDS",
        str(DEFAULT_MEMORY_RETRIEVAL_TIMEOUT_SECONDS),
    )
    try:
        return max(0.0, float(raw_value))
    except (TypeError, ValueError):
        logger.warning(
            "obs.memory.invalid_retrieval_timeout",
            value=raw_value,
            default_seconds=DEFAULT_MEMORY_RETRIEVAL_TIMEOUT_SECONDS,
        )
        return DEFAULT_MEMORY_RETRIEVAL_TIMEOUT_SECONDS


async def _get_episodic_memories_with_latency_budget(
    user: User,
    query: str,
    top_k: int,
    exclude_conversation_id: Optional[int],
):
    timeout_seconds = _memory_retrieval_timeout_seconds()
    lookup = get_episodic_memories_async(
        user,
        query,
        top_k=top_k,
        exclude_conversation_id=exclude_conversation_id,
    )
    if timeout_seconds <= 0:
        return await lookup
    try:
        return await asyncio.wait_for(lookup, timeout=timeout_seconds)
    except TimeoutError:
        logger.warning(
            "obs.memory.episodic_injection_timeout",
            timeout_seconds=timeout_seconds,
        )
        return []
    except Exception as exc:
        logger.warning(
            "obs.memory.episodic_injection_error",
            error_type=type(exc).__name__,
        )
        return []


def augment_conversation_with_memory(
    convo: 'Conversation',
    user: User,
    base_system: str,
    exclude_conversation_id: Optional[int] = None,
) -> None:
    """
    Set convo.system to base_system + user profile facts + retrieved episodic memories.
    Call before each LLM turn so episodic retrieval uses the latest user message.
    Uses base_system (e.g. db_convo.system_prompt) so we don't double-append when called multiple times.
    """
    profile_facts = get_user_profile_facts(user)
    query = get_last_user_message_text(convo)
    episodic = get_episodic_memories(user, query, top_k=EPISODIC_TOP_K, exclude_conversation_id=exclude_conversation_id)
    profile_facts = get_user_profile_facts(user)
    logger.info(
        "obs.memory.injection",
        profile_fact_count=len(profile_facts),
        episodic_memory_count=len(episodic),
    )
    block = format_memories_for_system(profile_facts, episodic, max_chars=_remaining_memory_chars(convo, base_system))
    convo.system = (base_system or "").rstrip() + block


async def augment_conversation_with_memory_async(
    convo: 'Conversation',
    user: User,
    base_system: str,
    exclude_conversation_id: Optional[int] = None,
    *,
    include_episodic: bool = True,
) -> None:
    """
    Like augment_conversation_with_memory but overlaps profile ORM load with async Mem0/pgvector episodic fetch.
    Prefer this from Channels/WebSocket handlers to reduce wall-clock latency before the LLM call.
    """
    query = get_last_user_message_text(convo)
    profile_task = database_sync_to_async(get_user_profile_facts)(user)
    if include_episodic:
        episodic_task = _get_episodic_memories_with_latency_budget(
            user,
            query,
            top_k=EPISODIC_TOP_K,
            exclude_conversation_id=exclude_conversation_id,
        )
        profile_facts, episodic = await asyncio.gather(profile_task, episodic_task)
    else:
        profile_facts = await profile_task
        episodic = []
    # A concurrent settings change may finish while episodic I/O is pending.
    # Re-read before formatting rather than injecting the earlier profile copy.
    profile_facts = await database_sync_to_async(get_user_profile_facts)(user)
    logger.info(
        "obs.memory.injection_async",
        profile_fact_count=len(profile_facts),
        episodic_memory_count=len(episodic),
        episodic_lookup_skipped=not include_episodic,
    )
    block = format_memories_for_system(profile_facts, episodic, max_chars=_remaining_memory_chars(convo, base_system))
    convo.system = (base_system or "").rstrip() + block


def _remaining_memory_chars(convo, base_system):
    from apps.chat.consumers.utils import CHAT_MAX_TOKENS
    from lib.llm.utils.prompt_budget import remaining_memory_chars
    messages = [message.model_dump(mode='json') for message in convo.messages]
    return remaining_memory_chars(system_text=base_system or '', message_dicts=messages, max_tokens=CHAT_MAX_TOKENS)


def create_episodic_memories_for_conversation(db_convo) -> None:
    """
    After saving a conversation, create EpisodicMemory rows for any assistant messages
    that don't have one yet. Content = previous user (or tool) message + assistant reply.
    Skips assistant messages that are tool-only (no text content).
    """
    messages = list(
        db_convo.db_messages.order_by('sequence_number').values(
            'sequence_number', 'role', 'content', 'message_uuid'
        )
    )
    prev_content = ""
    source_message_uuid = None
    for m in messages:
        role = m['role']
        content = m['content'] or ''
        msg_uuid = m['message_uuid']
        if role == 'user' or (role == 'tool' and content):
            prev_content = content
            source_message_uuid = str(msg_uuid) if role == 'user' else None
            continue
        if role != 'assistant':
            continue
        if not content or content == "** Empty Message, tool call **":
            prev_content = ""
            continue
        if not prev_content:
            continue
        if EpisodicMemory.objects.filter(
            user=db_convo.owner_id,
            assistant_message_uuid=msg_uuid,
        ).exists():
            prev_content = ""
            continue
        user_excerpt = prev_content[:500]
        assistant_excerpt = content[:500]
        memory_content = f"User: {user_excerpt}\nAssistant: {assistant_excerpt}"
        from apps.memory.services.profile_facts import content_digest
        digest = content_digest(prev_content)
        if use_mem0() and source_message_uuid is not None:
            _add_mem0_memory(
                user=db_convo.owner,
                user_content=prev_content,
                assistant_content=assistant_excerpt,
                conversation_id=db_convo.id,
                assistant_message_uuid=str(msg_uuid),
                source_message_uuid=source_message_uuid, source_content_digest=digest,
            )
        elif not use_mem0() and source_message_uuid is not None:
            _enqueue_profile_fact_promotion(user_id=db_convo.owner_id, user_content=prev_content,
                assistant_content='', source_conversation_id=db_convo.id, source_message_uuid=source_message_uuid,
                source_content_digest=digest)
        if (not use_mem0()) or MEM0_DUAL_WRITE_LOCAL:
            try:
                EpisodicMemory.objects.create(
                    user=db_convo.owner,
                    content=memory_content,
                    conversation=db_convo,
                    assistant_message_uuid=msg_uuid,
                )
            except IntegrityError as exc:
                if not _is_duplicate_episodic_memory_error(exc):
                    raise
                logger.info(
                    "obs.memory.duplicate_episodic_skipped",
                )
        prev_content = ""
