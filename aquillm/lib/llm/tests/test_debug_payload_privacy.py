"""DEBUG must retain operational metrics without publishing model payloads."""

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from lib.llm.providers import claude, complete_turn, openai
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import UserMessage

ANSWER = "ANSWER-SENTINEL-cerulean-otter"
SOURCE = "SOURCE-SENTINEL-selected-private-text"
TOOL = "TOOL-SENTINEL-private-argument"


@pytest.fixture(autouse=True)
def retain_provider_logs(caplog):
    # The application logger intentionally does not propagate to pytest's root
    # logger. Add a recording handler without removing any existing handler.
    logger = logging.getLogger("lib.llm.providers.openai")
    logger.addHandler(caplog.handler)
    try:
        yield
    finally:
        logger.removeHandler(caplog.handler)


def provider_for(name, *, tool):
    if name == "openai":
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=ANSWER,
                        tool_calls=[
                            SimpleNamespace(
                                id="call-1",
                                function=SimpleNamespace(
                                    name="lookup", arguments='{"query":"' + TOOL + '"}'
                                ),
                            )
                        ]
                        if tool
                        else None,
                    ),
                    finish_reason="tool_calls" if tool else "stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=17, completion_tokens=9),
        )
        create = AsyncMock(return_value=response)
        provider = openai.OpenAIInterface(
            SimpleNamespace(
                base_url="http://local.test",
                chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
            ),
            "test",
        )
    else:
        content = [SimpleNamespace(text=ANSWER)]
        if tool:
            content.append(
                SimpleNamespace(id="call-1", name="lookup", input={"query": TOOL})
            )
        create = AsyncMock(
            return_value=SimpleNamespace(
                content=content,
                stop_reason="tool_use" if tool else "end_turn",
                usage=SimpleNamespace(input_tokens=17, output_tokens=9),
            )
        )
        provider = claude.ClaudeInterface(
            SimpleNamespace(messages=SimpleNamespace(create=create)),
            model_override="test",
        )
    return provider, create


def enable_debug(monkeypatch, settings):
    settings.DEBUG = True
    # Exercise the existing import-time switches as enabled; after removal this
    # remains harmless and prevents the test from passing via DEBUG=False.
    for module in (openai, claude, complete_turn):
        monkeypatch.setattr(module, "DEBUG", True, raising=False)
    monkeypatch.setenv("OPENAI_STREAM_RESPONSES", "0")
    monkeypatch.setenv("OPENAI_CONTEXT_LIMIT", "16000")


def assert_private(capfd, caplog):
    captured = capfd.readouterr()
    observed = captured.out + captured.err + caplog.text
    for sentinel in (ANSWER, SOURCE, TOOL):
        assert sentinel not in observed
    return observed


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["openai", "claude"])
async def test_debug_provider_preserves_response_and_tool_without_dump(
    name,
    monkeypatch,
    settings,
    capfd,
    caplog,
):
    enable_debug(monkeypatch, settings)
    provider, create = provider_for(name, tool=True)
    response = await provider.get_message(
        system="System", messages=[{"role": "user", "content": SOURCE}], max_tokens=128
    )
    assert create.await_count == 1
    assert create.await_args.kwargs["messages"][-1]["content"] == SOURCE
    assert response.text == ANSWER
    assert response.tool_call == {
        "tool_call_id": "call-1",
        "tool_call_name": "lookup",
        "tool_call_input": {"query": TOOL},
    }
    assert (response.input_usage, response.output_usage) == (17, 9)
    observed = assert_private(capfd, caplog)
    if name == "openai":
        assert "obs.llm.request_completed" in observed


@pytest.mark.asyncio
async def test_debug_ordinary_turn_preserves_answer_and_final_stream_without_dump(
    monkeypatch,
    settings,
    capfd,
    caplog,
):
    enable_debug(monkeypatch, settings)
    provider, create = provider_for("openai", tool=False)
    conversation = Conversation(system="System", messages=[UserMessage(content=SOURCE)])
    events = []

    async def stream(event):
        events.append(event)

    updated, changed = await complete_turn.complete_conversation_turn(
        provider, conversation, max_tokens=128, stream_func=stream
    )
    assert changed == "changed" and create.await_count == 1
    assert updated[-1].content == ANSWER and updated[-1].usage == 26
    assert events and any(ANSWER in str(event) for event in events)
    assert "obs.llm.request_completed" in assert_private(capfd, caplog)
