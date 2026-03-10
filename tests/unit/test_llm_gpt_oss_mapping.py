import asyncio
from types import SimpleNamespace

from aquillm.llm import OpenAIInterface, Conversation, AssistantMessage, UserMessage, gpt_enc


class FakeOpenAICompletionsAPI:
    # this fakes the openai chat completions api

    def __init__(self, response):
        self.response = response
        self.last_create_kwargs = None

    async def create(self, **kwargs):
        # this stores the request args so the test can inspect them

        self.last_create_kwargs = kwargs
        return self.response


class FakeOpenAIChatAPI:

    def __init__(self, response):
        self.completions = FakeOpenAICompletionsAPI(response)


class FakeOpenAIClient:
    # this gives the interface a reg chat api

    def __init__(self, response):
        self.chat = FakeOpenAIChatAPI(response)


def test_openai_transform_tools_produces_openai_function_schema():
    # this checks internal tool definitions get converted to openai function tools

    response = SimpleNamespace()
    client = FakeOpenAIClient(response)
    iface = OpenAIInterface(client, model="gpt-oss:20b")

    transformed = asyncio.run(
        iface._transform_tools(
            [
                {
                    "name": "search_docs",
                    "description": "searches documents",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "the search query"}
                        },
                    },
                }
            ]
        )
    )

    assert transformed[0]["type"] == "function"
    assert transformed[0]["function"]["name"] == "search_docs"
    assert transformed[0]["function"]["description"] == "searches documents"
    assert transformed[0]["function"]["parameters"]["type"] == "object"
    assert transformed[0]["function"]["parameters"]["properties"]["query"]["type"] == "string"
    assert transformed[0]["function"]["parameters"]["required"] == ["query"]
    assert transformed[0]["function"]["strict"] is True


def test_openai_get_message_maps_text_only_response():
    # this checks plain text responses map into our standard response object

    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="gpt oss says hello", tool_calls=None),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=14, completion_tokens=6),
    )
    client = FakeOpenAIClient(response)
    iface = OpenAIInterface(client, model="gpt-oss:20b")

    result = asyncio.run(
        iface.get_message(
            system="developer instructions",
            messages=[{"role": "user", "content": "hi"}],
            messages_pydantic=[],
            max_tokens=200,
        )
    )

    assert result.text == "gpt oss says hello"
    assert result.tool_call == {}
    assert result.stop_reason == "stop"
    assert result.input_usage == 14
    assert result.output_usage == 6
    assert result.model == "gpt-oss:20b"

    sent_messages = client.chat.completions.last_create_kwargs["messages"]
    assert sent_messages[0] == {
        "role": "developer",
        "content": "developer instructions",
    }
    assert sent_messages[1] == {"role": "user", "content": "hi"}


def test_openai_get_message_maps_tool_call_response():
    # this checks openai style tool calls are converted into our standard fields

    tool_call = SimpleNamespace(
        id="call-123",
        function=SimpleNamespace(
            name="search_docs",
            arguments='{"query": "adapter tests"}',
        ),
    )
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=None, tool_calls=[tool_call]),
                finish_reason="tool_calls",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=19, completion_tokens=8),
    )
    client = FakeOpenAIClient(response)
    iface = OpenAIInterface(client, model="gpt-oss:20b")

    result = asyncio.run(
        iface.get_message(
            system="developer instructions",
            messages=[{"role": "user", "content": "search adapter tests"}],
            messages_pydantic=[],
            max_tokens=200,
        )
    )

    assert result.text is None
    assert result.tool_call["tool_call_id"] == "call-123"
    assert result.tool_call["tool_call_name"] == "search_docs"
    assert result.tool_call["tool_call_input"] == {"query": "adapter tests"}
    assert result.stop_reason == "tool_calls"
    assert result.model == "gpt-oss:20b"


def test_openai_token_count_uses_latest_assistant_usage_plus_new_message():
    # this checks that token_count uses the last assistant usage as the running total

    response = SimpleNamespace()
    client = FakeOpenAIClient(response)
    iface = OpenAIInterface(client, model="gpt-oss:20b")

    convo = Conversation(
        system="count tokens",
        messages=[
            UserMessage(content="first question"),
            AssistantMessage(content="first answer", stop_reason="end_turn", usage=40),
            UserMessage(content="second question"),
            AssistantMessage(content="second answer", stop_reason="end_turn", usage=75),
        ],
    )

    count = asyncio.run(iface.token_count(convo, new_message="next question"))
    expected = 75 + len(gpt_enc.encode("next question"))

    assert count == expected