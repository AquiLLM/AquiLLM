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
import structlog
from typing import TYPE_CHECKING, Optional

from channels.db import database_sync_to_async
from django.contrib.auth.models import User
from django.db import IntegrityError
from pgvector.django import L2Distance

from .models import UserMemoryFact, EpisodicMemory
from .utils import get_embedding

# Import from lib/memory for pure Python operations
from lib.memory import (
    RetrievedEpisodicMemory,
    EPISODIC_TOP_K,
    EPISODIC_MEMORY_MAX_CHARS,
    MEM0_DUAL_WRITE_LOCAL,
    clean_stable_facts,
    use_mem0,
    search_mem0_episodic_memories,
    search_mem0_episodic_memories_async,
    add_mem0_memory_with_client,
    extract_stable_facts,
    heuristic_facts_from_turn,
    has_remember_intent,
    normalize_remember_fact,
    format_memories_for_system,
)

if TYPE_CHECKING:
    from .llm import Conversation

logger = structlog.stdlib.get_logger(__name__)

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


def _categorize_profile_fact(fact: str) -> str:
    """Map a durable fact into the closest existing profile-memory category."""
    lowered = (fact or "").strip().lower()
    if not lowered:
        return "general"
    if lowered.startswith(("i prefer", "i like", "i want", "i need")):
        return "preference"
    if any(token in lowered for token in ("we use", "our stack", "project", "memory", "tool", "database", "qdrant", "memgraph")):
        return "project"
    if any(token in lowered for token in ("tone", "style", "concise", "verbose")):
        return "tone"
    if any(token in lowered for token in ("goal", "working on", "building", "trying to")):
        return "goals"
    return "general"


def _promote_profile_facts(user: User, facts: list[str]) -> None:
    """Persist durable facts into local profile memory for prompt injection."""
    for fact in facts:
        normalized = (fact or "").strip()
        if not normalized:
            continue
        UserMemoryFact.objects.get_or_create(
            user=user,
            fact=normalized,
            defaults={"category": _categorize_profile_fact(normalized)},
        )


def promote_profile_facts_for_turn(
    user_id: int,
    user_content: str,
    assistant_content: str,
) -> int:
    """Extract and persist durable facts for a completed turn."""
    user = User.objects.filter(id=user_id).first()
    if user is None:
        logger.warning("Skipping profile fact promotion; user_id=%s was not found.", user_id)
        return 0

    facts = extract_stable_facts(user_content, assistant_content)
    facts = clean_stable_facts(list(dict.fromkeys(facts)))
    if not facts and has_remember_intent(user_content):
        remembered = normalize_remember_fact(user_content)
        if remembered:
            facts = [remembered]
            logger.info("Using direct remember fallback; storing user content as memory fact.")

    if not facts:
        facts = heuristic_facts_from_turn(user_content, assistant_content)
        if facts:
            logger.info("Using heuristic memory extraction fallback; extracted %d fact(s).", len(facts))

    facts = clean_stable_facts(list(dict.fromkeys(facts)))
    if facts:
        _promote_profile_facts(user, facts)
    return len(facts)


def _enqueue_profile_fact_promotion(
    user_id: int,
    user_content: str,
    assistant_content: str,
) -> None:
    """Send durable-fact promotion to a lower-priority worker queue."""
    try:
        from .tasks import promote_profile_facts_task

        promote_profile_facts_task.delay(
            user_id=user_id,
            user_content=user_content,
            assistant_content=assistant_content,
        )
    except Exception as exc:
        logger.warning(
            "Failed to queue deferred profile fact promotion for user_id=%s; running inline. Error: %s",
            user_id,
            exc,
        )
        fact_count = promote_profile_facts_for_turn(
            user_id=user_id,
            user_content=user_content,
            assistant_content=assistant_content,
        )
        logger.info(
            "Inline profile fact promotion fallback completed with %d fact(s).",
            fact_count,
        )


def _add_mem0_memory(
    user: User,
    user_content: str,
    assistant_content: str,
    conversation_id: int,
    assistant_message_uuid: str,
) -> None:
    """Queue profile promotion separately and send the raw turn to Mem0 intelligent infer."""
    _enqueue_profile_fact_promotion(
        user_id=user.id,
        user_content=user_content,
        assistant_content=assistant_content,
    )

    if add_mem0_memory_with_client(
        user_id=str(user.id),
        user_content=user_content,
        assistant_content=assistant_content,
        conversation_id=conversation_id,
        assistant_message_uuid=assistant_message_uuid,
    ):
        logger.info("Mem0 intelligent write succeeded; profile fact promotion was deferred.")
        return

    logger.warning("Mem0 intelligent write produced no ADD events; profile fact promotion was deferred.")


def get_user_profile_facts(user: User):
    """Return all profile facts for the user (tone, goals, project, etc.)."""
    return list(UserMemoryFact.objects.filter(user=user).order_by('category', 'created_at'))


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
            return mem0_results
    return _get_episodic_memories_pgvector(
        user, query.strip(), top_k, exclude_conversation_id
    )


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
            return mem0_results
    return await database_sync_to_async(_get_episodic_memories_pgvector)(
        user, query.strip(), top_k, exclude_conversation_id
    )


def get_last_user_message_text(convo: 'Conversation') -> str:
    """Get the content of the most recent user message for use as retrieval query."""
    from .llm import UserMessage

    for msg in reversed(convo.messages):
        if isinstance(msg, UserMessage):
            return msg.content or ""
    return ""


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
    logger.info(
        "Memory injection: user_id=%s query=%r profile_facts=%d episodic_memories=%d",
        user.id,
        query[:180] if isinstance(query, str) else "",
        len(profile_facts),
        len(episodic),
    )
    block = format_memories_for_system(profile_facts, episodic)
    convo.system = (base_system or "").rstrip() + block


async def augment_conversation_with_memory_async(
    convo: 'Conversation',
    user: User,
    base_system: str,
    exclude_conversation_id: Optional[int] = None,
) -> None:
    """
    Like augment_conversation_with_memory but overlaps profile ORM load with async Mem0/pgvector episodic fetch.
    Prefer this from Channels/WebSocket handlers to reduce wall-clock latency before the LLM call.
    """
    query = get_last_user_message_text(convo)
    profile_task = database_sync_to_async(get_user_profile_facts)(user)
    episodic_task = get_episodic_memories_async(
        user, query, top_k=EPISODIC_TOP_K, exclude_conversation_id=exclude_conversation_id
    )
    profile_facts, episodic = await asyncio.gather(profile_task, episodic_task)
    logger.info(
        "Memory injection (async): user_id=%s query=%r profile_facts=%d episodic_memories=%d",
        user.id,
        query[:180] if isinstance(query, str) else "",
        len(profile_facts),
        len(episodic),
    )
    block = format_memories_for_system(profile_facts, episodic)
    convo.system = (base_system or "").rstrip() + block


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
    for m in messages:
        role = m['role']
        content = (m['content'] or "").strip()
        msg_uuid = m['message_uuid']
        if role == 'user' or (role == 'tool' and content):
            prev_content = content
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
        if use_mem0():
            _add_mem0_memory(
                user=db_convo.owner,
                user_content=user_excerpt,
                assistant_content=assistant_excerpt,
                conversation_id=db_convo.id,
                assistant_message_uuid=str(msg_uuid),
            )
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
                    "Skipping duplicate episodic memory insert after race: user_id=%s assistant_message_uuid=%s",
                    db_convo.owner_id,
                    msg_uuid,
                )
        prev_content = ""
