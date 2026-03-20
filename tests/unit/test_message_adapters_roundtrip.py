import pytest
from aquillm.llm import UserMessage, AssistantMessage, ToolMessage
from aquillm.message_adapters import (
    pydantic_message_to_django,
    django_message_to_pydantic,
)

# enable database access for these tests
pytestmark = pytest.mark.django_db


def test_user_message_roundtrip(user_message, db_conversation):
    # change a pydantic user message to a django row
    row = pydantic_message_to_django(user_message, db_conversation, seq_num=0)

    # change the row back to a pydantic message
    restored = django_message_to_pydantic(row)

    # verify message type survived the roundtrip
    assert isinstance(restored, UserMessage)

    # verify common fields are preserved
    assert restored.role == user_message.role
    assert restored.content == user_message.content
    assert restored.rating == user_message.rating
    assert restored.feedback_text == user_message.feedback_text
    assert restored.message_uuid == user_message.message_uuid


def test_assistant_message_roundtrip(assistant_message, db_conversation):
    # change assistant message to django model
    row = pydantic_message_to_django(assistant_message, db_conversation, seq_num=1)

    # change back to pydantic message
    restored = django_message_to_pydantic(row)

    # verify correct message type
    assert isinstance(restored, AssistantMessage)

    # verify common fields
    assert restored.role == assistant_message.role
    assert restored.content == assistant_message.content
    assert restored.rating == assistant_message.rating
    assert restored.feedback_text == assistant_message.feedback_text
    assert restored.message_uuid == assistant_message.message_uuid

    # verify assistant specific fields
    assert restored.model == assistant_message.model
    assert restored.stop_reason == assistant_message.stop_reason
    assert restored.tool_call_id == assistant_message.tool_call_id
    assert restored.tool_call_name == assistant_message.tool_call_name
    assert restored.tool_call_input == assistant_message.tool_call_input
    assert restored.usage == assistant_message.usage


def test_tool_message_roundtrip(tool_message, db_conversation):
    # change tool message to django model
    row = pydantic_message_to_django(tool_message, db_conversation, seq_num=2)

    # change back to pydantic message
    restored = django_message_to_pydantic(row)

    # verify type
    assert isinstance(restored, ToolMessage)

    # verify common fields
    assert restored.role == tool_message.role
    assert restored.content == tool_message.content
    assert restored.rating == tool_message.rating
    assert restored.feedback_text == tool_message.feedback_text
    assert restored.message_uuid == tool_message.message_uuid

    # verify tool specific fields
    assert restored.tool_name == tool_message.tool_name
    assert restored.arguments == tool_message.arguments
    assert restored.for_whom == tool_message.for_whom
    assert restored.result_dict == tool_message.result_dict