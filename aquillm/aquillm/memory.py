"""
User memory: profile facts (stable preferences) + episodic semantic memory (retrieved past exchanges).

- UserMemoryFact: injected into system on every chat for this user.
- EpisodicMemory: embedded past turns; we retrieve top-k by similarity to current message and inject.
"""

from typing import TYPE_CHECKING, Optional

from django.contrib.auth.models import User
from pgvector.django import L2Distance

from .models import UserMemoryFact, EpisodicMemory
from .utils import get_embedding

if TYPE_CHECKING:
    from .llm import Conversation

# How many past exchanges to retrieve for context
EPISODIC_TOP_K = 5


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
    qs = EpisodicMemory.objects.filter(user=user).exclude(embedding__isnull=True)
    if exclude_conversation_id is not None:
        qs = qs.exclude(conversation_id=exclude_conversation_id)
    try:
        embedding = get_embedding(query.strip(), input_type='search_query')
        return list(qs.order_by(L2Distance('embedding', embedding))[:top_k])
    except Exception:
        return []


def format_memories_for_system(profile_facts, episodic_memories) -> str:
    """Format profile facts and retrieved episodic memories as a block to append to the system prompt."""
    parts = []
    if profile_facts:
        lines = ["[User context / preferences you should respect]"]
        for f in profile_facts:
            lines.append(f"  - {f.fact}")
        parts.append("\n".join(lines))
    if episodic_memories:
        lines = ["[Relevant past exchanges with this user]"]
        for m in episodic_memories:
            lines.append(f"  - {m.content}")
        parts.append("\n".join(lines))
    if not parts:
        return ""
    return "\n\n" + "\n\n".join(parts)


def get_last_user_message_text(convo: 'Conversation') -> str:
    """Get the content of the most recent user message (or user-facing tool result) for use as retrieval query."""
    from .llm import UserMessage, ToolMessage

    for msg in reversed(convo.messages):
        if isinstance(msg, UserMessage):
            return msg.content or ""
        if isinstance(msg, ToolMessage) and msg.for_whom == 'assistant':
            # Tool result going to the model — still part of "user" context; use as query
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
    block = format_memories_for_system(profile_facts, episodic)
    convo.system = (base_system or "").rstrip() + block


def create_episodic_memories_for_conversation(db_convo) -> None:
    """
    After saving a conversation, create EpisodicMemory rows for any assistant messages
    that don't have one yet. Content = previous user (or tool) message + assistant reply.
    Skips assistant messages that are tool-only (no text content).
    """
    from .models import Message, WSConversation

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
        # Skip tool-only assistant messages (no substantive text)
        if not content or content == "** Empty Message, tool call **":
            prev_content = ""
            continue
        if not prev_content:
            continue
        # Dedupe: already have a memory for this assistant message?
        if EpisodicMemory.objects.filter(
            user=db_convo.owner_id,
            assistant_message_uuid=msg_uuid,
        ).exists():
            prev_content = ""
            continue
        memory_content = f"User: {prev_content[:500]}\nAssistant: {content[:500]}"
        EpisodicMemory.objects.create(
            user=db_convo.owner,
            content=memory_content,
            conversation=db_convo,
            assistant_message_uuid=msg_uuid,
        )
        prev_content = ""
