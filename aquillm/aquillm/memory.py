"""
User memory: profile facts (stable preferences) + episodic semantic memory (retrieved past exchanges).

- UserMemoryFact: injected into system on every chat for this user.
- EpisodicMemory: embedded past turns; we retrieve top-k by similarity to current message and inject.

Optional backend:
- MEM0 ("MEMORY_BACKEND=mem0"): use Mem0 for episodic retrieval/write with local fallback.

This module integrates lib/memory (pure Python) with Django models.
"""

import structlog
from typing import TYPE_CHECKING, Optional

from django.contrib.auth.models import User
from pgvector.django import L2Distance

from .models import UserMemoryFact, EpisodicMemory
from .utils import get_embedding

# Import from lib/memory for pure Python operations
from lib.memory import (
    RetrievedEpisodicMemory,
    EPISODIC_TOP_K,
    EPISODIC_MEMORY_MAX_CHARS,
    MEM0_DUAL_WRITE_LOCAL,
    use_mem0,
    search_mem0_episodic_memories,
    add_mem0_raw_facts,
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
    'format_memories_for_system',
    'get_last_user_message_text',
    'augment_conversation_with_memory',
    'create_episodic_memories_for_conversation',
]


def _add_mem0_memory(
    user: User,
    user_content: str,
    assistant_content: str,
    conversation_id: int,
    assistant_message_uuid: str,
) -> None:
    """Write memory to Mem0 with fact extraction."""
    facts = extract_stable_facts(user_content, assistant_content)
    facts = list(dict.fromkeys(facts))
    if not facts and has_remember_intent(user_content):
        remembered = normalize_remember_fact(user_content)
        if remembered:
            facts = [remembered]
            logger.info("obs.memory.direct_remember")

    if not facts:
        facts = heuristic_facts_from_turn(user_content, assistant_content)
        if facts:
            logger.info("obs.memory.heuristic_extract", fact_count=len(facts))

    if facts and add_mem0_raw_facts(
        user_id=str(user.id),
        facts=facts,
        conversation_id=conversation_id,
        assistant_message_uuid=assistant_message_uuid,
    ):
        logger.info("obs.memory.write_success", fact_count=len(facts))
        return

    if facts:
        logger.warning("obs.memory.write_no_events", fact_count=len(facts))
    else:
        logger.info("obs.memory.no_facts")

    add_mem0_memory_with_client(
        user_id=str(user.id),
        user_content=user_content,
        assistant_content=assistant_content,
        conversation_id=conversation_id,
        assistant_message_uuid=assistant_message_uuid,
    )


def get_user_profile_facts(user: User):
    """Return all profile facts for the user (tone, goals, project, etc.)."""
    return list(UserMemoryFact.objects.filter(user=user).order_by('category', 'created_at'))


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
    qs = EpisodicMemory.objects.filter(user=user).exclude(embedding__isnull=True)
    if exclude_conversation_id is not None:
        qs = qs.exclude(conversation_id=exclude_conversation_id)
    try:
        embedding = get_embedding(query.strip(), input_type='search_query')
        return list(qs.order_by(L2Distance('embedding', embedding))[:top_k])
    except Exception:
        return []


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
        "obs.memory.injection",
        user_id=user.id,
        query=query[:180] if isinstance(query, str) else "",
        profile_facts=len(profile_facts),
        episodic_memories=len(episodic),
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
            EpisodicMemory.objects.create(
                user=db_convo.owner,
                content=memory_content,
                conversation=db_convo,
                assistant_message_uuid=msg_uuid,
            )
        prev_content = ""
