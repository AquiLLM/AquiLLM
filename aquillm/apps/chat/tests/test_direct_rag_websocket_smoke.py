"""WebSocket smoke test for the direct RAG append path.

Exercises ``handle_chat_receive`` (same code path as ``ChatConsumer.receive``)
with ``RAG_DIRECT_ENABLED=1`` to verify that an append action triggers backend
retrieval *before* any LLM tool-selection spin.

Manual live WebSocket smoke (full stack, optional):

1. Set ``RAG_DIRECT_ENABLED=1`` in the repo-root ``.env``.
2. Start the dev stack::

       docker compose -f deploy/compose/base.yml -f deploy/compose/development.yml up web db redis qdrant

3. Open the chat UI, select one or more collections, and send a document question
   (e.g. "search the selected documents for calibration notes").
4. Confirm the assistant answers with citations and that retrieval happened without
   an initial tool-selection round trip (check server logs for
   ``direct_rag_turn_handled`` rather than ``llm_if.spin()`` on the append).

Run this file locally (PowerShell, repo root)::

    Get-Content .env | ForEach-Object {
        if ($_ -match '^\\s*([^#][^=]+)=(.*)$') {
            [Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim(), 'Process')
        }
    }
    cd aquillm
    python -m pytest apps/chat/tests/test_direct_rag_websocket_smoke.py -q --tb=short

No Postgres required: tests mock the consumer DB layer and exercise the same
``handle_chat_receive`` path as the live WebSocket.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from apps.chat.consumers.chat import ChatConsumer, CollectionsRef
from apps.chat.consumers.chat_receive import handle_chat_receive
from apps.chat.services import rag_pipeline
from apps.chat.tests.chat_message_test_support import (
    _FakeLLMInterface,
    _test_document_ids,
)
from aquillm.llm import Conversation
from lib.llm.types.messages import AssistantMessage, ToolMessage
from lib.llm.types.response import LLMResponse


def _results_payload() -> dict:
    return {
        "result": [
            {
                "rank": 1,
                "chunk_id": 1,
                "doc_id": "doc-a",
                "title": "Paper A",
                "text": "Calibration uses flat fields and dark frames.",
                "citation": "[doc:doc-a chunk:1]",
            }
        ],
        "retrieval_status": "results_found",
        "retrieved_count": 1,
        "retrieved_documents": ["Paper A"],
    }


def _append_payload(content: str, collections: list) -> str:
    return json.dumps(
        {
            "action": "append",
            "message": {"role": "user", "content": content},
            "collections": collections,
        }
    )


def _consumer(user, db_convo, llm_if) -> ChatConsumer:
    consumer = ChatConsumer()
    consumer.send = AsyncMock()
    consumer.scope = {"user": user, "url_route": {"kwargs": {"convo_id": db_convo.id}}}
    consumer.user = user
    consumer.db_convo = db_convo
    consumer.convo = Conversation(system="sys", messages=[])
    consumer.dead = False
    consumer.col_ref = CollectionsRef([])
    consumer.doc_tools = [_test_document_ids]
    consumer.tools = [_test_document_ids]
    consumer.last_sent_sequence = -1
    consumer.llm_if = llm_if
    consumer._send_stream_payload = AsyncMock()
    consumer._save_conversation = AsyncMock()
    return consumer


def _mock_user_and_convo():
    user = MagicMock()
    user.id = 1
    db_convo = MagicMock()
    db_convo.id = 42
    db_convo.selected_collection_ids = []
    db_convo.save = MagicMock()
    return user, db_convo


@pytest.mark.asyncio
@patch("apps.chat.consumers.chat_delta.aclose_old_connections", new_callable=AsyncMock)
@patch(
    "apps.chat.consumers.chat_receive.effective_base_system_for_memory_async",
    new_callable=AsyncMock,
    return_value="sys",
)
@patch("apps.chat.consumers.chat.enqueue_conversation_memories_task")
@patch("apps.chat.consumers.chat_receive.augment_conversation_with_memory_async", new_callable=AsyncMock)
@patch("apps.chat.consumers.chat_receive.run_llm_spin", new_callable=AsyncMock)
async def test_append_direct_rag_skips_tool_selection_spin(
    mock_spin,
    _augment,
    _mem_task,
    _memory_system,
    _aclose,
    monkeypatch,
):
    """Append with RAG_DIRECT_ENABLED=1 retrieves before LLM tool selection."""
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    order: list[str] = []
    raw = _results_payload()

    def fake_search(consumer, query, top_k):
        order.append("retrieval")
        return raw

    monkeypatch.setattr(rag_pipeline, "_run_vector_search", fake_search)

    answer = (
        "The paper describes calibration using flat fields and dark frames "
        "[doc:doc-a chunk:1]."
    )
    llm_if = _FakeLLMInterface(
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
    original_get_message = llm_if.get_message

    async def tracked_get_message(*args, **kwargs):
        order.append("get_message")
        return await original_get_message(*args, **kwargs)

    llm_if.get_message = tracked_get_message

    user, db_convo = _mock_user_and_convo()
    consumer = _consumer(user, db_convo, llm_if)

    await handle_chat_receive(
        consumer,
        _append_payload(
            "search the selected documents for calibration notes",
            [1],
        ),
    )

    mock_spin.assert_not_called()
    assert order.index("retrieval") < order.index("get_message")
    assert order.count("get_message") == 1

    messages = consumer.convo.messages
    tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
    assistant_msgs = [m for m in messages if isinstance(m, AssistantMessage)]
    assert any(m.tool_name == "vector_search" for m in tool_msgs)
    assert assistant_msgs[-1].content
    assert "calibration" in assistant_msgs[-1].content.lower()
    assert "[doc:doc-a chunk:1]" in assistant_msgs[-1].content


@pytest.mark.asyncio
@patch(
    "apps.chat.consumers.chat_receive.effective_base_system_for_memory_async",
    new_callable=AsyncMock,
    return_value="sys",
)
@patch("apps.chat.consumers.chat.enqueue_conversation_memories_task")
@patch("apps.chat.consumers.chat_receive.augment_conversation_with_memory_async", new_callable=AsyncMock)
@patch("apps.chat.consumers.chat_receive.run_llm_spin", new_callable=AsyncMock)
async def test_append_direct_rag_disabled_falls_back_to_spin(
    mock_spin,
    _augment,
    _mem_task,
    _memory_system,
    monkeypatch,
):
    """When RAG_DIRECT_ENABLED=0, append still uses the normal LLM spin path."""
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "0")

    user, db_convo = _mock_user_and_convo()
    llm_if = _FakeLLMInterface([])
    consumer = _consumer(user, db_convo, llm_if)

    await handle_chat_receive(
        consumer,
        _append_payload(
            "search the selected documents for calibration notes",
            [1],
        ),
    )

    mock_spin.assert_called_once()


@pytest.mark.asyncio
@patch("apps.chat.consumers.chat.build_memory_tools", return_value=[])
@patch("apps.chat.consumers.chat.build_document_tools", return_value=[])
@patch("apps.chat.consumers.chat.build_astronomy_tools", return_value=[])
@patch(
    "apps.chat.consumers.chat.effective_base_system_for_memory_async",
    new_callable=AsyncMock,
    return_value="sys",
)
@patch(
    "apps.chat.consumers.chat.augment_conversation_with_memory_async",
    new_callable=AsyncMock,
)
@patch("apps.chat.consumers.chat.run_llm_spin", new_callable=AsyncMock)
@patch(
    "apps.chat.consumers.chat.run_direct_rag_turn",
    new_callable=AsyncMock,
    return_value="handled",
)
async def test_reconnect_pending_user_turn_uses_direct_rag_before_tool_spin(
    mock_direct,
    mock_spin,
    _augment,
    _memory_system,
    _astronomy_tools,
    _document_tools,
    _memory_tools,
    monkeypatch,
):
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    pending = Conversation(
        system="sys",
        messages=[{"role": "user", "content": "what is attensity"}],
    )
    user, db_convo = _mock_user_and_convo()
    db_convo.selected_collection_ids = [203]
    db_convo.system_prompt = "sys"

    consumer = ChatConsumer()
    consumer.scope = {
        "user": user,
        "url_route": {"kwargs": {"convo_id": db_convo.id}},
    }
    consumer.accept = AsyncMock()
    consumer.send = AsyncMock()
    consumer._save_conversation = AsyncMock()
    consumer._ChatConsumer__get_all_user_collections = AsyncMock()
    consumer._ChatConsumer__get_convo = AsyncMock(return_value=db_convo)

    async def direct_sets_answer(*args, **kwargs):
        consumer.convo = pending + [
            AssistantMessage(
                content="Reconnected answer [doc:doc-a chunk:1].",
                stop_reason="end_turn",
            )
        ]
        return "handled"

    mock_direct.side_effect = direct_sets_answer

    with patch(
        "apps.chat.consumers.chat.load_conversation_from_db",
        return_value=pending,
    ), patch(
        "apps.chat.consumers.chat.database_sync_to_async",
        side_effect=lambda function: (
            lambda *args, **kwargs: _await_value(function(*args, **kwargs))
        ),
    ):
        await consumer.connect()

    mock_direct.assert_awaited_once()
    mock_spin.assert_not_awaited()


async def _await_value(value):
    return value
