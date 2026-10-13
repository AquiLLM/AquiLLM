"""Positive model validation must return the exact typed values, including tool turns."""
import pytest
from pydantic import ValidationError

from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import AssistantMessage, ToolMessage, UserMessage
from lib.llm.types.tools import ToolChoice
from aquillm.llm import message_to_user


def test_valid_message_model_validation_returns_instances_and_content():
    user = UserMessage.model_validate({"content": "question"})
    assistant = AssistantMessage.model_validate({"content": "answer", "stop_reason": "stop"})
    assert isinstance(user, UserMessage) and user.content == "question"
    assert isinstance(assistant, AssistantMessage) and assistant.content == "answer"
    assert Conversation.model_validate({"system": "base", "messages": [user, assistant]}).messages == [user, assistant]


def test_valid_complete_tool_turn_preserves_roles_and_tool_pair():
    user = UserMessage(content="question", tools=[message_to_user], tool_choice=ToolChoice(type="auto"))
    call = AssistantMessage(content="", stop_reason="tool_use", tool_call_id="call-1", tool_call_name="message_to_user")
    result = ToolMessage(tool_name="message_to_user", content="result", for_whom="assistant")
    final = AssistantMessage(content="answer", stop_reason="stop")
    convo = Conversation(system="base", messages=[user, call, result, final])
    assert convo.messages == [user, call, result, final]
    assert [message.role for message in convo] == ["user", "assistant", "tool", "assistant"]


def test_assistant_name_without_tool_id_is_rejected():
    with pytest.raises(ValidationError, match="tool_call_id and tool_call_name"):
        AssistantMessage(content="", stop_reason="tool_use", tool_call_name="message_to_user")


def test_rebound_pending_tool_call_keeps_a_valid_capability_pair():
    call = AssistantMessage(content="", stop_reason="tool_use", tool_call_id="call-1", tool_call_name="message_to_user")
    convo = Conversation(system="base", messages=[call])
    convo.rebind_tools([message_to_user])
    assert call.tools == [message_to_user]
    assert call.tool_choice == ToolChoice(type="auto")
    assert AssistantMessage.model_validate(call) is call
