import pytest
from aquillm.llm import Conversation
from aquillm.message_adapters import save_conversation_to_db, build_frontend_conversation_json

# allow database use
pytestmark = pytest.mark.django_db


def test_frontend_json_contains_common_fields_for_all_messages(
    db_conversation,
    user_message,
    assistant_message,
    tool_message,
):
    # build conversation with all message types
    convo = Conversation(
        system="Frontend system prompt",
        messages=[user_message, assistant_message, tool_message],
    )

    # persist conversation to database
    save_conversation_to_db(convo, db_conversation)

    # build json payload returned to frontend
    payload = build_frontend_conversation_json(db_conversation)

    # verify system prompt preserved
    assert payload["system"] == "Frontend system prompt"

    # verify all messages returned
    assert len(payload["messages"]) == 3

    # verify common fields included
    for msg in payload["messages"]:
        assert "role" in msg
        assert "content" in msg
        assert "message_uuid" in msg
        assert "rating" in msg
        assert isinstance(msg["message_uuid"], str)


def test_frontend_json_includes_assistant_fields_when_present(
    db_conversation,
    assistant_message,
):
    # create conversation with assistant message
    convo = Conversation(system="Assistant JSON test", messages=[assistant_message])

    save_conversation_to_db(convo, db_conversation)

    payload = build_frontend_conversation_json(db_conversation)
    msg = payload["messages"][0]

    # verify assistant tool metadata included
    assert msg["role"] == "assistant"
    assert msg["tool_call_name"] == "search_docs"
    assert msg["tool_call_input"] == {"query": "adapter tests"}
    assert msg["usage"] == 42


def test_frontend_json_omits_empty_assistant_optional_fields(db_conversation):
    from aquillm.models import Message
    import uuid

    # create assistant message without tool call metadata
    Message.objects.create(
        conversation=db_conversation,
        message_uuid=uuid.UUID("ffffffff-ffff-ffff-ffff-ffffffffffff"),
        role="assistant",
        content="Assistant without tool call",
        sequence_number=0,
        usage=0,
        tool_call_name=None,
        tool_call_input=None,
    )

    payload = build_frontend_conversation_json(db_conversation)
    msg = payload["messages"][0]

    # verify empty optional fields are not included
    assert msg["role"] == "assistant"
    assert "tool_call_name" not in msg
    assert "tool_call_input" not in msg
    assert "usage" not in msg


def test_frontend_json_includes_tool_fields(
    db_conversation,
    tool_message,
):
    # store conversation with tool message
    convo = Conversation(system="Tool JSON test", messages=[tool_message])

    save_conversation_to_db(convo, db_conversation)

    payload = build_frontend_conversation_json(db_conversation)
    msg = payload["messages"][0]

    # verify tool metadata included
    assert msg["role"] == "tool"
    assert msg["tool_name"] == "search_docs"
    assert msg["result_dict"] == {"result": "done"}
    assert msg["for_whom"] == "assistant"