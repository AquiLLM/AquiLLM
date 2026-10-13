"""Rebinding an authorized pending tool preserves its existing explicit choice."""
from aquillm.llm import message_to_user
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import AssistantMessage
from lib.llm.types.tools import ToolChoice


def test_rebinding_preserves_existing_any_tool_choice():
    choice = ToolChoice(type="any")
    call = AssistantMessage(content="", stop_reason="tool_use",
        tool_call_id="call-1", tool_call_name="message_to_user",
        tools=[message_to_user], tool_choice=choice)
    convo = Conversation(system="base", messages=[call])
    convo.rebind_tools([message_to_user])
    assert call.tools == [message_to_user]
    assert call.tool_choice is choice
    assert AssistantMessage.model_validate(call) is call
