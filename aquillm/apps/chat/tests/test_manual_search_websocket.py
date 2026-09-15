"""Slash commands retain direct retrieval through chat receive and reconnect."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from apps.chat.consumers.chat_receive import handle_chat_receive
from apps.chat.services import manual_search_turn
from apps.chat.tests.chat_message_test_support import _FakeLLMInterface
from apps.chat.tests.test_direct_rag_websocket_smoke import (
    _append_payload,
    _consumer,
    _mock_user_and_convo,
    _results_payload,
)
from lib.llm.types.messages import ToolMessage, UserMessage
from lib.llm.types.response import LLMResponse

DOC = "11111111-1111-4111-8111-111111111111"


@pytest.mark.parametrize(
    "command,tool_name",
    [
        ("/collection calibration", "vector_search"),
        (f"/search [{DOC}] calibration", "search_single_document"),
    ],
)
async def test_websocket_command_retrieves_before_model_and_publishes_final_delta(
    monkeypatch,
    command,
    tool_name,
):
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "0")
    order = []
    answer = "The paper describes calibration [doc:doc-a chunk:1]."
    llm = _FakeLLMInterface(
        [
            LLMResponse(
                text=answer,
                tool_call=None,
                stop_reason="end_turn",
                input_usage=1,
                output_usage=1,
            )
        ]
    )
    original = llm.get_message

    async def get_message(**kwargs):
        order.append("model")
        return await original(**kwargs)

    def search(**kwargs):
        order.append("search")
        assert kwargs["search_string"] == "calibration"
        if tool_name == "search_single_document":
            assert kwargs["doc_id"] == DOC
        return _results_payload()

    llm.get_message = get_message
    monkeypatch.setattr(manual_search_turn, tool_name + "_tool", lambda *args: search)
    user, db = _mock_user_and_convo()
    consumer = _consumer(user, db, llm)
    with (
        patch(
            "apps.chat.consumers.chat_receive.augment_conversation_with_memory_async",
            new=AsyncMock(),
        ),
        patch(
            "apps.chat.consumers.chat_receive.effective_base_system_for_memory_async",
            new=AsyncMock(return_value="sys"),
        ),
        patch("apps.chat.consumers.chat_receive.run_llm_spin", new=AsyncMock()) as spin,
    ):
        await handle_chat_receive(consumer, _append_payload(command, [1]))
    assert order == ["search", "model"]
    spin.assert_not_called()
    assert consumer.convo.messages[0].content == command
    assert consumer.convo[-1].content.startswith(answer)
    assert isinstance(consumer.convo[-2], ToolMessage)
    assert consumer.convo[-2].tool_name == tool_name
    assert consumer.last_sent_sequence == len(consumer.convo) - 1
    assert any(
        call.kwargs.get("create_memories")
        for call in consumer._save_conversation.await_args_list
    )
    payloads = [
        json.loads(call.kwargs["text_data"]) for call in consumer.send.await_args_list
    ]
    assert any(
        any(
            answer in msg.get("content", "")
            for msg in p.get("delta", {}).get("messages", [])
        )
        for p in payloads
    )


async def test_reconnect_pending_manual_command_uses_requested_search(monkeypatch):
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "0")
    calls = []
    monkeypatch.setattr(
        manual_search_turn,
        "vector_search_tool",
        lambda *args: lambda **kwargs: calls.append(kwargs) or {"result": []},
    )
    user, db = _mock_user_and_convo()
    consumer = _consumer(user, db, _FakeLLMInterface([]))
    consumer.convo += UserMessage(content="/collection calibration")
    pending = consumer.convo
    db.selected_collection_ids = [1]
    db.system_prompt = "sys"
    consumer.accept = AsyncMock()
    consumer._ChatConsumer__get_all_user_collections = AsyncMock()
    consumer._ChatConsumer__get_convo = AsyncMock(return_value=db)
    with (
        patch(
            "apps.chat.consumers.chat.load_conversation_from_db", return_value=pending
        ),
        patch("apps.chat.consumers.chat.build_document_tools", return_value=[]),
        patch("apps.chat.consumers.chat.build_astronomy_tools", return_value=[]),
        patch(
            "apps.chat.consumers.chat.augment_conversation_with_memory_async",
            new=AsyncMock(),
        ),
        patch(
            "apps.chat.consumers.chat.effective_base_system_for_memory_async",
            new=AsyncMock(return_value="sys"),
        ),
        patch("apps.chat.consumers.chat.run_llm_spin", new=AsyncMock()) as spin,
    ):
        await consumer.connect()
    assert len(calls) == 1
    assert calls[0]["search_string"] == "calibration"
    assert "no relevant passages" in consumer.convo[-1].content
    spin.assert_not_called()
