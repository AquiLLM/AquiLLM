"""The normal research-assistant flow reuses its one assessment before selection."""
from hashlib import sha256
import json
from types import SimpleNamespace

import pytest

from apps.chat.refs import CollectionsRef
from apps.chat.tests.persisted_direct_test_support import close_direct_worker, persisted_direct_case
from apps.chat.tests.test_rag_preservation_routes import configure, fake_search
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import UserMessage

pytestmark = [pytest.mark.asyncio, pytest.mark.django_db(transaction=True),
              pytest.mark.usefixtures("close_direct_worker")]


@pytest.mark.parametrize("enabled", [False, True])
async def test_existing_assessment_guides_normal_selection_without_more_calls(monkeypatch, enabled):
    from apps.chat.services import rag_pipeline
    from apps.chat.services.rag_turn import preservation_turn
    from apps.chat.tests.chat_message_test_support import _FakeLLMInterface
    from apps.documents.services import chunk_rerank_selection_provider
    from apps.documents.services.chunk_rerank_window_adapter import WindowSelectionScorer
    from lib.llm.types.response import LLMResponse

    configure(monkeypatch, windowed=True)
    monkeypatch.setenv("RAG_ITERATIVE_RETRIEVAL_ENABLED", "1")
    monkeypatch.setenv("RAG_SELECTION_SUPPORT_HINTS_ENABLED", str(int(enabled)))
    monkeypatch.setenv("RAG_EVIDENCE_SELECTION_MODE", "adaptive")
    monkeypatch.setenv("RAG_DIRECT_TOP_K", "1")
    texts = ("The main result is faster restart after a controller failure.",
             "The condition is that the write-ahead journal remains intact.",
             "Administrative details describe the printing office.")
    case = await persisted_direct_case([
        dict(doc_id="study", chunk_id=i, title="Recovery study", text=text)
        for i, text in enumerate(texts, 1)])
    chunks = tuple(case.chunks.values())
    fake_search(monkeypatch)

    class Scorer:
        def score_pair(self, pair, timeout_seconds):
            score = .95 if "main result" in pair[1] else .94 if "condition" in pair[1] else .1
            return score, pair

    monkeypatch.setattr(chunk_rerank_selection_provider, "current_selection_scorer",
        lambda *, deadline, clock, turn_budget: WindowSelectionScorer(Scorer(), budget=turn_budget,
            deadline=deadline, clock=clock, pair_counter=lambda q, d: len(q) + len(d) + 8,
            scorer_identity="integration-hints"))
    assessment = json.dumps(dict(requested_aspects=["failure condition"], support=[dict(
        aspect="failure condition", chunk_id=chunks[1].pk,
        source_fingerprint=sha256(texts[1].encode()).hexdigest(), start=0, end=len(texts[1]))],
        unresolved_aspects=[], next_action=None))
    selected = chunks[1] if enabled else chunks[0]
    answer = ("Recovery requires an intact journal." if enabled else "The study reports a faster restart.")
    answer += f" [doc:{selected.doc_id} chunk:{selected.pk}]"
    llm = _FakeLLMInterface([LLMResponse(text=text, tool_call=None, stop_reason="end_turn",
        input_usage=1, output_usage=1) for text in (assessment, answer)])
    convo = Conversation(system="sys", messages=[UserMessage(content="Explain the recovery result and its condition.")])
    doc = case.documents["study"]
    consumer = SimpleNamespace(user=case.user, col_ref=CollectionsRef([doc.collection_id]), convo=convo)
    async with preservation_turn(consumer, max_func_calls=3) as runtime:
        assert await rag_pipeline.run_direct_rag_turn(consumer, llm, convo) == "handled"
        packet = runtime.observation["delivered_packet"]
        assert [row.chunk_id for row in packet.selection.candidates] == [selected.pk]
        assert packet.selection.score_status == "model"
        assert runtime.budget.actions_used == 1
        assert runtime.budget.planner_calls == 1
        assert runtime.budget.pairs_used["final"] == 3
        assert not getattr(runtime, "_selection_support_hints", None)
    assert len(llm.calls) == 2  # Existing planner and actual model-written synthesis.
    assert consumer.convo[-1].content.strip() == answer
