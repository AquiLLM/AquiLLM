"""Synthetic retrieval messages shared by direct source and legacy paths."""

import uuid

from lib.llm.providers import image_context as imgctx
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import AssistantMessage, ToolMessage, UserMessage


def _append_retrieval_messages(
    convo: Conversation, query: str, raw_result: dict, top_k: int
) -> Conversation:
    """Append a synthetic tool-call + tool-result so synthesis sees a post-tool turn."""
    arguments = {"search_string": query, "top_k": top_k}
    assistant_tool_call = AssistantMessage(
        content="",
        stop_reason="tool_use",
        tool_call_id=str(uuid.uuid4()),
        tool_call_name="vector_search",
        tool_call_input=arguments,
    )
    tool_message = ToolMessage(
        tool_name="vector_search",
        for_whom="assistant",
        content=imgctx.serialize_tool_result_for_llm(raw_result),
        arguments=arguments,
        result_dict=raw_result,
    )
    return convo + [assistant_tool_call, tool_message]


def _latest_user_message(convo: Conversation) -> UserMessage | None:
    if len(convo) == 0:
        return None
    last = convo[-1]
    return last if isinstance(last, UserMessage) else None


def _has_prior_vector_search(convo: Conversation) -> bool:
    """Return whether this conversation contains a reusable retrieval query."""
    return any(
        getattr(message, "tool_call_name", None) == "vector_search"
        and isinstance(getattr(message, "tool_call_input", None), dict)
        and bool(message.tool_call_input.get("search_string"))
        for message in convo.messages[:-1]
    )
