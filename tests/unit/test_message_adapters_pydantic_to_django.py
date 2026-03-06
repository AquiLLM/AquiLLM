import pytest
from aquillm.message_adapters import pydantic_message_to_django
from aquillm.models import Message

pytestmark = pytest.mark.django_db


def test_user_message_maps_to_django_row(user_message, db_conversation):
    msg = pydantic_message_to_django(user_message, db_conversation, seq_num=0)

    assert isinstance(msg, Message)
    assert msg.conversation == db_conversation
    assert msg.message_uuid == user_message.message_uuid
    assert msg.role == "user"
    assert msg.content == "Hello from user"
    assert msg.rating == 5
    assert msg.feedback_text == "helpful"
    assert msg.sequence_number == 0

    assert msg.model is None
    assert msg.stop_reason is None
    assert msg.tool_call_id is None
    assert msg.tool_call_name is None
    assert msg.tool_call_input is None
    assert msg.usage == 0

    assert msg.tool_name is None
    assert msg.arguments is None
    assert msg.for_whom is None
    assert msg.result_dict is None


def test_assistant_message_maps_to_django_row(assistant_message, db_conversation):
    msg = pydantic_message_to_django(assistant_message, db_conversation, seq_num=1)

    assert isinstance(msg, Message)
    assert msg.conversation == db_conversation
    assert msg.message_uuid == assistant_message.message_uuid
    assert msg.role == "assistant"
    assert msg.content == "Hello from assistant"
    assert msg.rating == 4
    assert msg.feedback_text == "good answer"
    assert msg.sequence_number == 1

    assert msg.model == "gpt-4o"
    assert msg.stop_reason == "end_turn"
    assert msg.tool_call_id == "tool-call-1"
    assert msg.tool_call_name == "search_docs"
    assert msg.tool_call_input == {"query": "adapter tests"}
    assert msg.usage == 42

    assert msg.tool_name is None
    assert msg.arguments is None
    assert msg.for_whom is None
    assert msg.result_dict is None


def test_tool_message_maps_to_django_row(tool_message, db_conversation):
    msg = pydantic_message_to_django(tool_message, db_conversation, seq_num=2)

    assert isinstance(msg, Message)
    assert msg.conversation == db_conversation
    assert msg.message_uuid == tool_message.message_uuid
    assert msg.role == "tool"
    assert msg.content == "{'result': 'done'}"
    assert msg.rating == 3
    assert msg.feedback_text == "tool output"
    assert msg.sequence_number == 2

    assert msg.tool_name == "search_docs"
    assert msg.arguments == {"query": "adapter tests"}
    assert msg.for_whom == "assistant"
    assert msg.result_dict == {"result": "done"}

    assert msg.model is None
    assert msg.stop_reason is None
    assert msg.tool_call_id is None
    assert msg.tool_call_name is None
    assert msg.tool_call_input is None
    assert msg.usage == 0