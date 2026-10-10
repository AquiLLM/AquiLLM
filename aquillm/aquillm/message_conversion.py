"""Convert runtime messages and stored rows without persisting or publishing."""

from lib.llm.providers.visibility import sanitize_assistant_text

from .llm import AssistantMessage, LLM_Message, ToolMessage, UserMessage
from .models import Message, WSConversation


def pydantic_message_to_django(
    msg: LLM_Message, conversation: WSConversation, seq_num: int
) -> Message:
    """Convert a Pydantic message to a Django Message instance (unsaved).

    Returns an unsaved Message object — the caller is responsible for saving it
    (typically via bulk_create for performance).
    """
    # Fields shared by all message types
    content = (
        sanitize_assistant_text(
            msg.content,
            allow_short_final=msg.stop_reason in ("stop", "end_turn")
            and not msg.tool_call_name,
        )
        if isinstance(msg, AssistantMessage)
        else msg.content
    )
    if getattr(msg, "_rag_answer", False):
        content = msg.content
    common = {
        "conversation": conversation,  # FK linking this message to its conversation
        "message_uuid": msg.message_uuid,  # frontend message identity
        "role": msg.role,  # 'user', 'assistant', or 'tool'
        "content": content,  # the actual message text
        "rating": msg.rating,  # user rating (1-5) or None
        "feedback_text": msg.feedback_text,  # optional user feedback text
        "sequence_number": seq_num,  # position in the conversation (0, 1, 2, ...)
    }

    # Add role-specific fields depending on message type
    if isinstance(msg, AssistantMessage):
        return Message(
            **common,
            model=msg.model,  # which LLM model generated this response
            stop_reason=msg.stop_reason,  # e.g. 'end_turn' or 'tool_use'
            tool_call_id=msg.tool_call_id,  # ID if the LLM called a tool
            tool_call_name=msg.tool_call_name,  # e.g. 'vector_search'
            tool_call_input=msg.tool_call_input,  # arguments passed to the tool
            usage=msg.usage,  # token count for this response
        )
    elif isinstance(msg, ToolMessage):
        return Message(
            **common,
            tool_name=msg.tool_name,  # which tool produced this result
            arguments=msg.arguments,  # arguments the tool was called with
            for_whom=msg.for_whom,  # who the result is for ('assistant' or 'user')
            result_dict=msg.result_dict,  # the tool's output data
        )
    else:
        # UserMessage — only needs the common fields
        return Message(**common)


def django_message_to_pydantic(msg: Message) -> LLM_Message:
    """Convert a Django Message row to a Pydantic message object.

    Used when loading a conversation from the database for runtime use.
    The Pydantic object can be passed to the LLM API or rendered for the frontend.
    """
    # Fields shared by all message types
    common = {
        "content": msg.content,
        "rating": msg.rating,
        "feedback_text": msg.feedback_text,
        "message_uuid": msg.message_uuid,
    }

    if msg.role == "assistant":
        return AssistantMessage(
            **common,
            model=msg.model,
            stop_reason=msg.stop_reason
            or "end_turn",  # default to 'end_turn' if not stored
            tool_call_id=msg.tool_call_id,
            tool_call_name=msg.tool_call_name,
            tool_call_input=msg.tool_call_input,
            usage=msg.usage,
        )
    elif msg.role == "tool":
        return ToolMessage(
            **common,
            tool_name=msg.tool_name
            or "",  # default to empty string (required by Pydantic)
            arguments=msg.arguments,
            for_whom=msg.for_whom
            or "assistant",  # default to 'assistant' (required by Pydantic)
            result_dict=msg.result_dict
            or {},  # default to empty dict (required by Pydantic)
        )
    else:
        return UserMessage(**common)
