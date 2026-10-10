"""Ordinary tools cannot revive historical document authority."""

import pytest

from lib.llm.providers.complete_turn import complete_conversation_turn
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import AssistantMessage, ToolMessage, UserMessage
from lib.llm.types.response import LLMResponse


@pytest.mark.parametrize("historical_citation", [False, True])
async def test_ordinary_tool_after_document_history_remains_ordinary(
    monkeypatch, historical_citation
):
    monkeypatch.setenv("LLM_STREAM_FINAL_ANSWER_ONLY", "0")
    answer = "The weather service reports clear skies."
    if historical_citation:
        answer = "An old document finding [doc:old chunk:1]."
    requests, events = [], []

    class Model:
        base_args = {}

        async def get_message(self, **kwargs):
            requests.append(kwargs)
            if callback := kwargs.get("stream_callback"):
                await callback(
                    {
                        "content": answer,
                        "done": True,
                        "message_uuid": kwargs["stream_message_uuid"],
                        "role": "assistant",
                        "stop_reason": "stop",
                        "usage": 1,
                    }
                )
            return LLMResponse(
                text=answer,
                tool_call={},
                stop_reason="stop",
                input_usage=1,
                output_usage=1,
            )

    convo = Conversation(
        system="s",
        messages=[
            ToolMessage(
                tool_name="vector_search",
                content="Old source",
                for_whom="assistant",
                result_dict={
                    "result": [{"doc_id": "old", "chunk_id": 1, "text": "Old source"}]
                },
            ),
            AssistantMessage(
                content="Old answer [doc:old chunk:1].", stop_reason="end_turn"
            ),
            UserMessage(content="What did the weather service return?"),
            ToolMessage(
                tool_name="weather",
                content="Clear skies",
                for_whom="assistant",
                result_dict={"result": "Clear skies"},
            ),
        ],
    )

    async def stream(payload):
        events.append(payload)

    result, _ = await complete_conversation_turn(
        Model(), convo, 1024, stream_func=stream
    )
    from lib.llm.answer_publication import LIMITED_ANSWER

    assert result[-1].content == (LIMITED_ANSWER if historical_citation else answer)
    assert result[-1]._rag_answer is historical_citation
    assert events and all(event["content"] == result[-1].content for event in events)
    if not historical_citation:
        assert len(requests) == 1
    assert all("doc:old" not in event.get("content", "") for event in events)
