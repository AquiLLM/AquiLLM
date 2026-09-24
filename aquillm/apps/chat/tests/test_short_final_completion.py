"""Completed short provider answers survive real completion and UI persistence."""

from types import SimpleNamespace

import pytest
from asgiref.sync import async_to_sync
from django.contrib.auth import get_user_model

from aquillm.message_adapters import (
    build_frontend_conversation_json,
    load_conversation_from_db,
    save_conversation_to_db,
)
from aquillm.models import WSConversation
from lib.llm.providers.complete_turn import complete_conversation_turn
from lib.llm.providers.openai_streaming import consume_streaming_completion
from lib.llm.providers.visibility import visible_stream_content
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import UserMessage
from lib.llm.types.response import LLMResponse


@pytest.mark.django_db
@pytest.mark.parametrize(
    "answer,stop", [("Yes", "end_turn"), ("42", "stop"), ("Approved", "end_turn")]
)
def test_provider_completion_stream_and_saved_short_final(answer, stop, monkeypatch):
    monkeypatch.setenv("LLM_STREAM_FINAL_ANSWER_ONLY", "1")
    payloads = []

    async def get_message(**kwargs):
        return LLMResponse(
            text=answer, tool_call=None, stop_reason=stop, input_usage=1, output_usage=1
        )

    async def send(payload):
        payloads.append(payload)

    user = get_user_model().objects.create_user(username="short-final")
    stored = WSConversation.objects.create(owner=user, system_prompt="base")
    conversation = Conversation(
        system="base", messages=[UserMessage(content="Answer briefly.")]
    )
    result, status = async_to_sync(complete_conversation_turn)(
        SimpleNamespace(base_args={}, get_message=get_message),
        conversation,
        max_tokens=64,
        stream_func=send,
    )
    assert status == "changed" and result[-1].content == answer
    assert payloads[-1]["done"] and payloads[-1]["content"] == answer
    save_conversation_to_db(result, stored)
    assert load_conversation_from_db(stored)[-1].content == answer
    assert build_frontend_conversation_json(stored)["messages"][-1]["content"] == answer


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stop,expected", [("stop", "Yes"), ("length", ""), (None, ""), ("tool_calls", "")]
)
async def test_actual_provider_stream_requires_terminal_reason(stop, expected):
    payloads = []

    async def stream():
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="Yes", tool_calls=[]),
                    finish_reason=stop,
                )
            ],
            usage=None,
        )

    async def send(payload):
        payloads.append(payload)

    await consume_streaming_completion(
        stream=stream(),
        stream_callback=send,
        stream_message_uuid="short",
        raw_tools=None,
        model_name="test",
    )
    final = [p["content"] for p in payloads if p["done"]]
    assert final == ([expected] if expected else [])


@pytest.mark.parametrize(
    "text", ["", " ", "Searching...", "<think>Yes</think>", "tool"]
)
def test_short_final_eligibility_does_not_expose_nonanswers(text):
    assert not visible_stream_content(
        text, raw_tools=None, done=True, stop_reason="stop"
    )
    assert not visible_stream_content(
        "Yes", raw_tools=None, done=True, stop_reason=None
    )
    assert not visible_stream_content(
        "Yes",
        raw_tools=None,
        done=True,
        stop_reason="stop",
        tool_call_payload={"name": "tool"},
    )
