"""Display-only metadata for incomplete calls loaded from retained DB rows."""
from copy import deepcopy
from dataclasses import dataclass
from uuid import UUID

from lib.llm.types.messages import AssistantMessage


@dataclass(frozen=True)
class RetainedToolHistory:
    message_uuid: UUID
    content: str
    runtime_stop_reason: str
    stored_stop_reason: str | None
    tool_call_id: str | None
    tool_call_name: str | None
    tool_call_input: dict | None


def capture_tool_history(message, row):
    """Called only by the stored-row adapter after ordinary typed validation."""
    message._retained_tool_history = RetainedToolHistory(
        message.message_uuid, row.content, message.stop_reason, row.stop_reason,
        row.tool_call_id, row.tool_call_name, deepcopy(row.tool_call_input),
    )


def retained_tool_history(message):
    """Never restore a retained call onto an edited or newly bound message."""
    if not isinstance(message, AssistantMessage):
        return None
    history = message._retained_tool_history
    if not isinstance(history, RetainedToolHistory):
        return None
    if (message.message_uuid != history.message_uuid
            or message.content != history.content
            or message.stop_reason != history.runtime_stop_reason
            or message.tools is not None or message.tool_choice is not None
            or any(value is not None for value in (
                message.tool_call_id, message.tool_call_name, message.tool_call_input))):
        return None
    return history
