import uuid
import pytest
from aquillm.llm import UserMessage, AssistantMessage, ToolMessage
from aquillm.message_adapters import django_message_to_pydantic
from aquillm.models import Message

# enable database access
pytestmark = pytest.mark.django_db


def test_user_row_maps_to_pydantic_message(db_conversation):
    # create a user message row
    row = Message(
        conversation=None,
        message_uuid=uuid.uuid4(),
        role="user",
        content="Hello",
        rating=5,
        feedback_text="great",
        sequence_number=0,
    )

    # convert row to pydantic message
    msg = django_message_to_pydantic(row)

    # verify correct message type
    assert isinstance(msg, UserMessage)

    # verify fields mapped correctly
    assert msg.role == "user"
    assert msg.content == "Hello"
    assert msg.rating == 5
    assert msg.feedback_text == "great"
    assert msg.message_uuid == row.message_uuid


def test_assistant_row_maps_to_pydantic_message(db_conversation):
    # create assistant message row with tool call metadata
    row = Message.objects.create(
        conversation=db_conversation,
        message_uuid=uuid.UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
        role="assistant",
        content="Assistant row content",
        rating=4,
        feedback_text="solid",
        sequence_number=1,
        model="gpt-4o",
        stop_reason="end_turn",
        tool_call_id="tc-1",
        tool_call_name="search_docs",
        tool_call_input={"query": "adapter"},
        usage=77,
    )

    # convert row to pydantic
    msg = django_message_to_pydantic(row)

    # verify mapping
    assert isinstance(msg, AssistantMessage)
    assert msg.role == "assistant"
    assert msg.content == "Assistant row content"
    assert msg.rating == 4
    assert msg.feedback_text == "solid"
    assert msg.message_uuid == row.message_uuid
    assert msg.model == "gpt-4o"
    assert msg.stop_reason == "end_turn"
    assert msg.tool_call_id == "tc-1"
    assert msg.tool_call_name == "search_docs"
    assert msg.tool_call_input == {"query": "adapter"}
    assert msg.usage == 77


def test_tool_row_maps_to_pydantic_message(db_conversation):
    # create tool message row
    row = Message.objects.create(
        conversation=db_conversation,
        message_uuid=uuid.UUID("cccccccc-cccc-cccc-cccc-cccccccccccc"),
        role="tool",
        content="Tool row content",
        rating=2,
        feedback_text="tool feedback",
        sequence_number=2,
        tool_name="search_docs",
        arguments={"query": "adapter"},
        for_whom="assistant",
        result_dict={"result": "done"},
    )

    # convert to pydantic
    msg = django_message_to_pydantic(row)

    # verify mapping
    assert isinstance(msg, ToolMessage)
    assert msg.role == "tool"
    assert msg.content == "Tool row content"
    assert msg.rating == 2
    assert msg.feedback_text == "tool feedback"
    assert msg.message_uuid == row.message_uuid
    assert msg.tool_name == "search_docs"
    assert msg.arguments == {"query": "adapter"}
    assert msg.for_whom == "assistant"
    assert msg.result_dict == {"result": "done"}


def test_assistant_row_defaults_stop_reason_to_end_turn(db_conversation):
    # create assistant message without stop reason
    row = Message.objects.create(
        conversation=db_conversation,
        message_uuid=uuid.UUID("dddddddd-dddd-dddd-dddd-dddddddddddd"),
        role="assistant",
        content="Assistant with no stop reason",
        sequence_number=3,
        stop_reason=None,
    )

    # convert to pydantic
    msg = django_message_to_pydantic(row)

    # verify safe default
    assert isinstance(msg, AssistantMessage)
    assert msg.stop_reason == "end_turn"


def test_tool_row_applies_safe_defaults_when_fields_missing(db_conversation):
    # create tool row with missing optional values
    row = Message.objects.create(
        conversation=db_conversation,
        message_uuid=uuid.UUID("eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee"),
        role="tool",
        content="Tool with missing optional fields",
        sequence_number=4,
        tool_name=None,
        arguments=None,
        for_whom=None,
        result_dict=None,
    )

    # convert to pydantic
    msg = django_message_to_pydantic(row)

    # verify fallback defaults
    assert isinstance(msg, ToolMessage)
    assert msg.tool_name == ""
    assert msg.arguments is None
    assert msg.for_whom == "assistant"
    assert msg.result_dict == {}