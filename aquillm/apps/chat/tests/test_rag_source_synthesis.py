"""Source-mode synthesis protects full exact text and returns explicit limits."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from apps.chat.services.rag_evidence import EvidencePacket
from apps.chat.services.rag_synthesis import synthesize_from_evidence
from lib.llm.answer_publication import LIMITED_ANSWER
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import ToolMessage, UserMessage

pytestmark = pytest.mark.django_db(transaction=True)


def inputs(*, text=None):
    row = {
        "doc_id": "11111111-1111-4111-8111-111111111111",
        "chunk_id": 1,
        "chunk": 0,
        "text": text if text is not None else "background " * 200 + "Tail exception: do not retain without consent.",
    }
    packet = EvidencePacket(
        [row], [], [], "consent", "selected", "results_found", "", 500, source_mode=True
    )
    from apps.chat.services.rag_selection_types import (
        EvidenceSelection,
        SelectionCandidate,
        SelectionProfile,
    )
    from lib.retrieval.evidence import fingerprint_source

    row["citation"] = f"[doc:{row['doc_id']} chunk:1]"
    packet.selection = EvidenceSelection(
        (
            SelectionCandidate(
                1,
                row["doc_id"],
                0,
                row["text"],
                1,
                1,
                fingerprint_source(row["text"]),
                row,
                token_cost=500,
            ),
        ),
        500,
        SelectionProfile("test", 1, 0, "v1"),
        "rank_fallback",
    )
    convo = Conversation(
        system="system",
        messages=[
            ToolMessage(
                tool_name="vector_search",
                for_whom="assistant",
                content="old evidence",
                arguments={},
                result_dict={},
            ),
            UserMessage(content="consent policy"),
            ToolMessage(
                tool_name="vector_search",
                for_whom="assistant",
                content="preview",
                arguments={},
                result_dict={},
            ),
        ],
    )
    return convo, packet


from apps.chat.tests.source_synthesis_authority_support import authorized_source_scope


@pytest.mark.asyncio
async def test_unknown_context_returns_limited_without_model_dispatch(monkeypatch):
    monkeypatch.setenv("OPENAI_CONTEXT_LIMIT", "0")
    monkeypatch.setenv("PROMPT_BUDGET_CONTEXT_LIMIT", "0")
    from lib.llm.evidence_guard import ContextLimited

    llm = SimpleNamespace(
        complete=AsyncMock(side_effect=ContextLimited("unknown_model_context"))
    )
    convo, packet = inputs()
    result = await synthesize_from_evidence(llm, convo, packet)
    assert result[-1].content == LIMITED_ANSWER
    assert "no relevant" not in result[-1].content.lower()
    llm.complete.assert_not_called()


@pytest.mark.asyncio
async def test_synthesis_guard_keeps_history_and_exact_payload(monkeypatch):
    from lib.llm.evidence_guard import current_protection
    from lib.llm.types.messages import AssistantMessage

    monkeypatch.setenv("OPENAI_CONTEXT_LIMIT", "16000")
    convo, packet = inputs()
    observed = []

    async def complete(request, *_a, **_kw):
        state = current_protection()
        assert state is not None
        assert state.payload == request[-1].content
        assert packet.chunks[0]["text"] in state.payload
        assert request[0].content != "old evidence"
        observed.append(state)
        return request + [
            AssistantMessage(content="Supported answer", stop_reason="end_turn")
        ], "changed"

    async with authorized_source_scope(packet, monkeypatch, convo):
        result = await synthesize_from_evidence(
            SimpleNamespace(complete=complete), convo, packet
        )
    assert result[0].content == convo[0].content == "old evidence"
    assert observed
    assert current_protection() is None


@pytest.mark.asyncio
async def test_last_moment_revalidation_can_remove_changed_packet(monkeypatch):
    from channels.db import database_sync_to_async

    monkeypatch.setenv("OPENAI_CONTEXT_LIMIT", "16000")
    convo, packet = inputs()
    llm = SimpleNamespace(complete=AsyncMock())
    async with authorized_source_scope(packet, monkeypatch, convo) as acquired:
        original = acquired.runtime._frozen_retrieval_scope
        await database_sync_to_async(acquired.permission.delete)()
        assert acquired.runtime._frozen_retrieval_scope is original
        result = await synthesize_from_evidence(llm, convo, packet)
    assert result[-1].content == LIMITED_ANSWER
    llm.complete.assert_not_called()
