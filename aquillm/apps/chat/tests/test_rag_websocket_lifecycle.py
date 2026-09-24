"""The actual connect/append owners retain one ledger across direct and spin."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from apps.chat.consumers.chat_receive import handle_chat_receive
from apps.chat.tests.test_direct_rag_websocket_smoke import (
    _append_payload,
    _consumer,
    _mock_user_and_convo,
)
from apps.documents.services.source_loading import SourceRuntime, current_source_runtime
from lib.llm.turn_context import current_turn
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import AssistantMessage, UserMessage


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["connect", "append"])
async def test_consumer_direct_fallback_share_budget_and_next_turn_is_new(
    entry, monkeypatch
):
    from apps.chat.services import rag_turn

    monkeypatch.setenv("RAG_EVIDENCE_TEXT_MODE", "source")
    monkeypatch.setattr(
        rag_turn, "create_runtime", lambda _, budget: SourceRuntime(budget, None)
    )
    user, db = _mock_user_and_convo()
    consumer = _consumer(user, db, object())
    captured = []

    async def direct(*args, **kwargs):
        budget = current_source_runtime().budget
        assert current_turn().budget is budget
        captured.append(budget)
        assert budget.reserve_action("initial")
        return "skipped"

    async def spin(*args, **kwargs):
        assert current_turn().budget is captured[-1]
        assert current_turn().budget.actions_used == 1
        consumer.convo += AssistantMessage(content="Final", stop_reason="end_turn")

    module = (
        "apps.chat.consumers.chat"
        if entry == "connect"
        else "apps.chat.consumers.chat_receive"
    )
    with (
        patch(module + ".run_direct_rag_turn", direct),
        patch(module + ".run_llm_spin", spin),
        patch(module + ".augment_conversation_with_memory_async", AsyncMock()),
        patch(
            module + ".effective_base_system_for_memory_async",
            AsyncMock(return_value="sys"),
        ),
    ):
        for _ in range(2):
            if entry == "append":
                await handle_chat_receive(
                    consumer, _append_payload("search selected documents", [1])
                )
            else:
                pending = Conversation(
                    system="sys",
                    messages=[UserMessage(content="Search selected documents")],
                )
                db.selected_collection_ids, db.system_prompt = [1], "sys"
                consumer.accept = AsyncMock()
                consumer._ChatConsumer__get_all_user_collections = AsyncMock()
                consumer._ChatConsumer__get_convo = AsyncMock(return_value=db)
                with (
                    patch(module + ".load_conversation_from_db", return_value=pending),
                    patch(module + ".build_document_tools", return_value=[]),
                    patch(module + ".build_astronomy_tools", return_value=[]),
                    patch(module + ".build_memory_tools", return_value=[]),
                ):
                    await consumer.connect()
            assert current_source_runtime() is current_turn() is None
            assert not captured[-1].can_publish()
    assert len(captured) == 2
    assert captured[0] is not captured[1]


@pytest.mark.asyncio
async def test_spin_cancellation_skips_unconditional_finalizer():
    from types import SimpleNamespace

    from apps.chat.consumers.chat_publish import run_llm_spin

    consumer = SimpleNamespace(
        convo=Conversation(system="sys", messages=[]), send=AsyncMock()
    )
    llm = SimpleNamespace(spin=AsyncMock(side_effect=asyncio.CancelledError))
    with pytest.raises(asyncio.CancelledError):
        await run_llm_spin(
            consumer,
            llm,
            consumer.convo,
            max_func_calls=3,
            max_tokens=100,
            send_func=AsyncMock(),
        )
    consumer.send.assert_not_called()
    assert not consumer._spin_active
