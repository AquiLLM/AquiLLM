"""Source-mode synthesis protects full exact text and returns explicit limits."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from apps.chat.services.rag_evidence import EvidencePacket
from apps.chat.services.rag_synthesis import synthesize_from_evidence
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import ToolMessage, UserMessage

pytestmark = pytest.mark.django_db(transaction=True)


def inputs():
    row = {
        "doc_id": "11111111-1111-4111-8111-111111111111",
        "chunk_id": 1,
        "chunk": 0,
        "text": "background " * 200 + "Tail exception: do not retain without consent.",
    }
    packet = EvidencePacket(
        [row], [], [], "consent", "selected", "results_found", "", 500, source_mode=True
    )
    from apps.chat.services.rag_selection_types import (
        EvidenceSelection,
        SelectionCandidate,
        SelectionProfile,
    )
    from apps.chat.tests.test_rag_selection_scoring import Policy, _authorization
    from lib.retrieval.evidence import fingerprint_source

    row["citation"] = f"[doc:{row['doc_id']} chunk:1]"
    packet.source_authorization = _authorization(Policy())
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


@contextmanager
def authorized_source_scope(packet, monkeypatch):
    from apps.chat.services import rag_source_hydration
    from apps.chat.tests.test_rag_selection_scoring import _chunk
    from apps.documents.services.source_loading import (
        SourceRuntime,
        source_runtime_scope,
    )
    from lib.retrieval.turn_budget import TurnBudget, TurnLimits

    chunk = _chunk(1, packet.chunks[0]["text"])
    monkeypatch.setattr(rag_source_hydration, "source_query_rows", lambda *_: (chunk,))
    with source_runtime_scope(
        SourceRuntime(TurnBudget(TurnLimits()), packet.source_authorization)
    ):
        yield


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
    assert "context" in result[-1].content.lower()
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

    with authorized_source_scope(packet, monkeypatch):
        result = await synthesize_from_evidence(
            SimpleNamespace(complete=complete), convo, packet
        )
    assert result[0].content == convo[0].content == "old evidence"
    assert observed
    assert current_protection() is None


@pytest.mark.asyncio
async def test_last_moment_revalidation_can_remove_changed_packet(monkeypatch):
    from apps.chat.services import rag_source_hydration
    from apps.chat.services.rag_selection_types import (
        EvidenceSelection,
        SelectionProfile,
    )

    monkeypatch.setenv("OPENAI_CONTEXT_LIMIT", "16000")
    convo, packet = inputs()
    packet.source_authorization = object()
    packet.selection = EvidenceSelection(
        (), 0, SelectionProfile("test", 1, 0, "v1"), "rank_fallback"
    )
    monkeypatch.setattr(
        rag_source_hydration, "revalidate_source_candidates", lambda *_a, **_kw: ()
    )
    llm = SimpleNamespace(complete=AsyncMock())
    with authorized_source_scope(packet, monkeypatch):
        result = await synthesize_from_evidence(llm, convo, packet)
    assert "limits" in result[-1].content
    llm.complete.assert_not_called()
