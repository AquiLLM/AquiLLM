import pytest
from pydantic import ValidationError

from aquillm.llm import (
    AssistantMessage,
    ClaudeInterface,
    Conversation,
    OpenAIInterface,
    ToolChoice,
    UserMessage,
    message_to_user,
)


def test_message_to_user_tool_definition_is_correct():
    # this checks the tool metadata that gets exposed to llms

    assert message_to_user.name == "message_to_user"
    assert message_to_user.for_whom == "user"

    schema = message_to_user.llm_definition["input_schema"]

    assert schema["type"] == "object"
    assert schema["required"] == ["message"]
    assert schema["properties"]["message"]["type"] == "string"
    assert (
        schema["properties"]["message"]["description"]
        == "The message to send to the user."
    )


def test_message_to_user_tool_executes():
    # this checks the wrapped function still runs correctly

    result = message_to_user(message="hello user")

    assert result == {"result": "hello user"}


def test_claude_interface_sets_model_override():
    # this checks we can build a claude interface with a chosen model

    iface = ClaudeInterface(anthropic_client=object(), model_override="claude-test-model")

    assert iface.client is not None
    assert iface.base_args["model"] == "claude-test-model"


def test_gpt_oss_interface_sets_model_name():
    # this checks gpt oss goes through the openai style interface

    iface = OpenAIInterface(openai_client=object(), model="gpt-oss:20b")

    assert iface.client is not None
    assert iface.base_args["model"] == "gpt-oss:20b"


def test_tool_choice_requires_name_for_tool_type():
    # this checks a specific tool choice must name the tool

    with pytest.raises(ValueError):
        ToolChoice(type="tool")


def test_tool_choice_rejects_name_for_non_tool_type():
    # this checks name is only allowed when type is tool

    with pytest.raises(ValueError):
        ToolChoice(type="auto", name="search_docs")


def test_user_message_rejects_tools_without_tool_choice():
    # this checks tools and tool_choice must be provided together

    with pytest.raises(ValidationError):
        UserMessage(
            content="please use tools",
            tools=[message_to_user],
        )


def test_user_message_rejects_tool_choice_without_tools():
    # this checks tool_choice alone is not valid

    with pytest.raises(ValidationError):
        UserMessage(
            content="please use tools",
            tool_choice=ToolChoice(type="auto"),
        )


def test_assistant_tool_call_requires_pair():
    # this checks tool call id and tool call name must come together

    with pytest.raises(ValidationError):
        AssistantMessage(
            content="calling a tool",
            stop_reason="tool_use",
            tool_call_id="tool-call-1",
        )


@pytest.mark.parametrize("role", ["user", "assistant"])
def test_conversation_preserves_adjacent_historical_messages(role):
    # Interrupted provider work can leave a durable user turn before the next append.
    # Loading history must preserve every original message and feedback identity.
    cls = UserMessage if role == "user" else AssistantMessage
    fields = {} if role == "user" else {"stop_reason": "end_turn"}
    messages = [cls(content="first", **fields), cls(content="second", **fields)]
    convo = Conversation(system="test system", messages=messages)
    assert convo.messages == messages
    assert [message.message_uuid for message in convo] == [message.message_uuid for message in messages]
