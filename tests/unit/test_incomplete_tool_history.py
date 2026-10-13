"""Retained malformed calls remain lossless history, never pending execution."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from asgiref.sync import async_to_sync
from pydantic import ValidationError

from aquillm.llm import AssistantMessage, LLMTool, OpenAIInterface, ToolChoice, UserMessage
from aquillm.message_adapters import (
    build_frontend_conversation_json, load_conversation_from_db,
    pydantic_message_to_frontend_dict, save_conversation_to_db,
)
from aquillm.models import Message
from aquillm.message_conversion import pydantic_message_to_django
from apps.chat.consumers.chat import ChatConsumer

pytestmark = pytest.mark.django_db
PAIRS = [(None, "lookup"), ("old-call", None)]
FIELDS = ("message_uuid", "role", "content", "model", "stop_reason", "tool_call_id",
          "tool_call_name", "tool_call_input", "usage", "rating", "feedback_text",
          "feedback_submitted_at", "sequence_number")


def _tool(calls):
    return LLMTool(llm_definition={"name": "lookup", "description": "lookup",
        "input_schema": {"type": "object", "properties": {}}},
        for_whom="assistant", _function=lambda: calls.append("executed") or {"result": "ok"})


def _stored(db_conversation, pair):
    return Message.objects.create(conversation=db_conversation, role="assistant",
        content="<think>retained private reasoning</think>Saved assistant history.",
        model="historical-model", stop_reason=None, tool_call_id=pair[0],
        tool_call_name=pair[1], tool_call_input={"literal": "retained argument"},
        usage=37, rating=4, feedback_text="keep this feedback", sequence_number=0)


@pytest.mark.parametrize("pair", PAIRS)
def test_incomplete_persisted_call_roundtrip_is_display_only(db_conversation, pair):
    row = _stored(db_conversation, pair)
    original = Message.objects.filter(pk=row.pk).values(*FIELDS).get()
    convo = load_conversation_from_db(db_conversation)
    message = convo[-1]
    assert isinstance(message, AssistantMessage)
    assert message.tool_call_id is message.tool_call_name is message.tool_call_input is None
    assert message.message_uuid == row.message_uuid
    assert message.content == row.content
    assert message.rating == 4 and message.feedback_text == "keep this feedback"
    assert not ChatConsumer._has_pending_turn(SimpleNamespace(convo=convo))
    calls = []
    convo.rebind_tools([_tool(calls)])
    assert convo.messages == [message] and not message.tools and not message.tool_choice
    assert not ChatConsumer._has_pending_turn(SimpleNamespace(convo=convo))
    llm = OpenAIInterface(openai_client=object(), model="synthetic-history-test")
    llm.get_message = AsyncMock(side_effect=AssertionError("no provider for historical call"))
    llm.call_tool = Mock(side_effect=AssertionError("no tool for historical call"))
    unchanged, status = async_to_sync(llm.complete)(convo, 128)
    assert unchanged is convo and status == "unchanged"
    llm.get_message.assert_not_called()
    llm.call_tool.assert_not_called()
    assert calls == []
    frontend = build_frontend_conversation_json(db_conversation)["messages"][0]
    in_memory = pydantic_message_to_frontend_dict(message)
    for rendered in (frontend, in_memory):
        assert rendered["message_uuid"] == str(row.message_uuid)
        assert rendered["rating"] == 4 and rendered["usage"] == 37
        assert "retained private reasoning" not in rendered["content"]
        if pair[1]:
            assert rendered["tool_call_name"] == pair[1]
            assert rendered["tool_call_input"] == row.tool_call_input
            assert rendered["content"] == ""
        else:
            assert "tool_call_name" not in rendered
    # Public provider/model serialization must not expose private historical metadata.
    serialized = message.model_dump()
    assert serialized["tool_call_id"] is serialized["tool_call_name"] is None
    assert serialized["tool_call_input"] is None
    continued = convo + UserMessage(content="Continue with a new request.")
    save_conversation_to_db(continued, db_conversation)
    assert Message.objects.filter(pk=row.pk).values(*FIELDS).get() == original
    reloaded = load_conversation_from_db(db_conversation)
    assert len(reloaded) == 2 and reloaded[0].message_uuid == row.message_uuid
    assert reloaded[0].tool_call_id is reloaded[0].tool_call_name is None
    assert reloaded[0].content == row.content
    assert reloaded[1].message_uuid == continued[-1].message_uuid


@pytest.mark.parametrize("pair", PAIRS)
def test_new_incomplete_tool_messages_still_reject(pair):
    with pytest.raises(ValidationError, match="tool_call_id and tool_call_name"):
        AssistantMessage(content="new call", stop_reason="tool_use",
                         tool_call_id=pair[0], tool_call_name=pair[1])


def test_complete_persisted_tool_pair_remains_pending_and_executable(db_conversation):
    row = Message.objects.create(conversation=db_conversation, role="assistant",
        content="", stop_reason="tool_use", tool_call_id="complete-call",
        tool_call_name="lookup", tool_call_input={}, usage=2, sequence_number=0)
    convo = load_conversation_from_db(db_conversation)
    assert (convo[-1].tool_call_id, convo[-1].tool_call_name) == ("complete-call", "lookup")
    assert ChatConsumer._has_pending_turn(SimpleNamespace(convo=convo))
    calls = []
    convo.rebind_tools([_tool(calls)])
    llm = OpenAIInterface(openai_client=object(), model="synthetic-history-test")
    llm.get_message = AsyncMock(side_effect=AssertionError("tool path has no provider"))
    result, status = async_to_sync(llm.complete)(convo, 128)
    assert status == "changed" and calls == ["executed"]
    assert result[-1].result_dict == {"result": "ok"}
    llm.get_message.assert_not_called()
    frontend = build_frontend_conversation_json(db_conversation)["messages"][0]
    assert frontend["message_uuid"] == str(row.message_uuid)
    assert frontend["tool_call_name"] == "lookup" and frontend["tool_call_input"] == {}


@pytest.mark.parametrize("edit", ["content", "stop_reason", "message_uuid", "tools", "tool_pair"])
def test_changed_history_does_not_restore_old_tool_metadata(db_conversation, edit):
    _stored(db_conversation, (None, "lookup"))
    message = load_conversation_from_db(db_conversation)[-1]
    if edit == "content":
        message.content = "Explicitly edited historical text."
    elif edit == "stop_reason":
        message.stop_reason = "stop"
    elif edit == "message_uuid":
        message.message_uuid = uuid4()
    elif edit == "tools":
        message.tools, message.tool_choice = [_tool([])], ToolChoice(type="auto")
    else:
        message.tool_call_id, message.tool_call_name = "new-call", "new_lookup"
        message.tool_call_input = {"fresh": True}
    projected = pydantic_message_to_django(message, db_conversation, 0)
    rendered = pydantic_message_to_frontend_dict(message)
    if edit == "tool_pair":
        assert (projected.tool_call_id, projected.tool_call_name) == ("new-call", "new_lookup")
        assert projected.tool_call_input == {"fresh": True}
        assert rendered["tool_call_name"] == "new_lookup"
    else:
        assert projected.tool_call_id is projected.tool_call_name is projected.tool_call_input is None
        assert "tool_call_name" not in rendered
    assert projected.stop_reason == message.stop_reason
    assert projected.message_uuid == message.message_uuid
