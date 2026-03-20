import asyncio
from types import SimpleNamespace

from aquillm.llm import ClaudeInterface, Conversation, UserMessage


class FakeClaudeMessagesAPI:
    # this fakes the anthropic messages api

    def __init__(self, response, token_count_value=123):
        self.response = response
        self.token_count_value = token_count_value
        self.last_create_kwargs = None
        self.last_count_kwargs = None

    async def create(self, **kwargs):
        # this stores request args so the test can inspect them

        self.last_create_kwargs = kwargs
        return self.response

    async def count_tokens(self, **kwargs):
        # this stores token count args so the test can inspect them

        self.last_count_kwargs = kwargs
        return SimpleNamespace(input_tokens=self.token_count_value)


class FakeClaudeClient:
    # this gives the interface a messages object
    
    def __init__(self, response, token_count_value=123):
        self.messages = FakeClaudeMessagesAPI(response, token_count_value=token_count_value)


def test_claude_get_message_maps_text_only_response():
    # this checks plain text responses map into our standard response object

    response = SimpleNamespace(
        content=[SimpleNamespace(text="claude says hello")],
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=11, output_tokens=7),
    )
    client = FakeClaudeClient(response)
    iface = ClaudeInterface(client, model_override="claude-test-model")

    result = asyncio.run(
        iface.get_message(
            system="system text",
            messages=[{"role": "user", "content": "hi"}],
            messages_pydantic=[],
            max_tokens=200,
        )
    )

    assert result.text == "claude says hello"
    assert result.tool_call == {}
    assert result.stop_reason == "end_turn"
    assert result.input_usage == 11
    assert result.output_usage == 7
    assert result.model == "claude-test-model"

    assert "messages_pydantic" not in client.messages.last_create_kwargs
    assert client.messages.last_create_kwargs["system"] == "system text"
    assert client.messages.last_create_kwargs["messages"] == [
        {"role": "user", "content": "hi"}
    ]


def test_claude_get_message_maps_tool_call_response():
    # this checks tool calls are extracted from anthropic style content blocks

    response = SimpleNamespace(
        content=[
            SimpleNamespace(text="working on it"),
            SimpleNamespace(id="tool-123", name="search_docs", input={"query": "adapters"}),
        ],
        stop_reason="tool_use",
        usage=SimpleNamespace(input_tokens=21, output_tokens=9),
    )
    client = FakeClaudeClient(response)
    iface = ClaudeInterface(client, model_override="claude-test-model")

    result = asyncio.run(
        iface.get_message(
            system="system text",
            messages=[{"role": "user", "content": "search for adapters"}],
            messages_pydantic=[],
            max_tokens=200,
        )
    )

    assert result.text == "working on it"
    assert result.tool_call["tool_call_id"] == "tool-123"
    assert result.tool_call["tool_call_name"] == "search_docs"
    assert result.tool_call["tool_call_input"] == {"query": "adapters"}
    assert result.stop_reason == "tool_use"
    assert result.model == "claude-test-model"


def test_claude_token_count_uses_rendered_messages_and_new_message():
    # this checks token counting sends the expected conversation shape

    response = SimpleNamespace(
        content=[SimpleNamespace(text="ok")],
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
    )
    client = FakeClaudeClient(response, token_count_value=321)
    iface = ClaudeInterface(client, model_override="claude-test-model")

    convo = Conversation(
        system="count these tokens",
        messages=[UserMessage(content="first question")],
    )

    count = asyncio.run(iface.token_count(convo, new_message="next message"))

    assert count == 321
    assert client.messages.last_count_kwargs["system"] == "count these tokens"
    assert client.messages.last_count_kwargs["messages"][0]["content"] == "first question"
    assert client.messages.last_count_kwargs["messages"][1]["content"] == "next message"