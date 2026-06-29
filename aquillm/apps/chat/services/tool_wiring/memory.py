"""Past-chat search LLM tool (Django-bound).

Wraps apps.chat.services.conversation_search.search_conversation_chunks (hybrid
vector + trigram retrieval over the user's indexed ConversationChunks) as an
assistant tool, so the model can deliberately recall things discussed in *other*
conversation threads.
"""
from __future__ import annotations

from django.contrib.auth.models import User

from aquillm.llm import LLMTool, ToolResultDict, llm_tool
from apps.chat.consumers.utils import truncate_tool_text
from apps.chat.refs import ChatRef


def _current_conversation_id(chat_ref: ChatRef):
    chat = getattr(chat_ref, "chat", None)
    db_convo = getattr(chat, "db_convo", None)
    return getattr(db_convo, "id", None)


def search_past_chats_tool(user: User, chat_ref: ChatRef) -> LLMTool:
    @llm_tool(
        for_whom="assistant",
        required=["query", "top_k"],
        param_descs={
            "query": (
                "Required. What to look for across this user's PAST conversations (other "
                "chat threads), e.g. a topic, decision, or detail discussed before."
            ),
            "top_k": (
                "Required. Integer from 1 to 15 — number of past-chat excerpts to return. "
                "Use 8 for a typical recall question."
            ),
        },
    )
    def search_past_chats(query: str, top_k: int = 8) -> ToolResultDict:
        """
        Semantically search this user's EARLIER conversations (separate chat threads)
        for things previously discussed. Use this when the user refers to a past chat
        or asks what was said, decided, or talked about before. This does NOT search
        documents—use vector_search for documents. Returns excerpts, each labeled with
        the title of the conversation it came from.
        """
        if not query.strip():
            return {"exception": "query must not be empty"}
        if top_k < 1 or top_k > 15:
            return {"exception": f"top_k must be between 1 and 15, got {top_k}"}

        from apps.chat.models import WSConversation
        from apps.chat.services.conversation_search import search_conversation_chunks

        chunks = search_conversation_chunks(
            user,
            query,
            top_k,
            exclude_conversation_id=_current_conversation_id(chat_ref),
        )
        if not chunks:
            return {"result": "No relevant past conversations found."}

        names_by_id = dict(
            WSConversation.objects.filter(
                owner=user, id__in={c.conversation_id for c in chunks}
            ).values_list("id", "name")
        )

        lines = []
        for i, chunk in enumerate(chunks, start=1):
            title = (names_by_id.get(chunk.conversation_id) or "").strip() or "Untitled conversation"
            excerpt = truncate_tool_text(chunk.content)
            lines.append(f"{i}. From chat \"{title}\" (id {chunk.conversation_id}):\n{excerpt}")

        return {
            "result": (
                f"Found {len(chunks)} relevant excerpt(s) from past conversations:\n\n"
                + "\n\n".join(lines)
            )
        }

    return search_past_chats


__all__ = ["search_past_chats_tool"]
