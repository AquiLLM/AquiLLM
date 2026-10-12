"""A scoring-only timeout preserves exact support without extending its allowance."""

from types import SimpleNamespace

import pytest
from apps.chat.tests.persisted_direct_test_support import close_direct_worker

from apps.chat.services.rag_selection_scoring import prepare_selection_candidates
from apps.chat.tests.test_rag_selection_scoring import (
    Policy,
    _authorization,
    _chunk,
    _pool,
)
from apps.documents.services.chunk_rerank_window_adapter import WindowSelectionScorer
from apps.documents.services.source_loading import SourcePreparationLimited
from lib.llm.evidence_guard import ContextLimited
from lib.retrieval.turn_budget import TurnBudget, TurnLimits


class Scenario:
    def __init__(self, *, stop=None):
        self.now = 0.0
        self.budget = TurnBudget(TurnLimits(), clock=lambda: self.now)
        self.policy = Policy()
        self.authorization = _authorization(self.policy)
        self.chunks = (
            _chunk(1, "Supported result is 42 mK only below 1 Pa."),
            _chunk(2, "The exception requires written consent."),
        )
        self.calls = []
        self.loads = 0
        self.stop = stop
        self.scorer = WindowSelectionScorer(
            SimpleNamespace(scorer_fingerprint="test", score_pair=self.score),
            budget=self.budget,
            pair_counter=lambda q, d: len(q) + len(d) + 8,
            deadline=3.0,
            clock=lambda: self.now,
        )

    def score(self, pair, timeout):
        self.calls.append((pair, timeout))
        self.now = 15.001 if self.stop == "global" else 3.001
        if self.stop == "cancel":
            self.budget.close("cancelled")
        elif self.stop == "revoke":
            self.policy.allowed = ()
        elif self.stop == "revise":
            self.chunks[0].content = "Revised source must not inherit the old support."
        return 1.0, pair

    def load(self, *_):
        self.loads += 1
        if self.stop == "initial":
            self.now = 3.001
        elif self.stop == "revalidation" and self.loads == 2:
            self.now = 15.001
        return self.chunks

    def prepare(self):
        return prepare_selection_candidates(
            pool=_pool(self.chunks),
            primary_query="result?",
            authorization=self.authorization,
            deadline=3.0,
            allow_new_scores=True,
            scorer=self.scorer,
            chunk_loader=self.load,
            turn_budget=self.budget,
            source_mode=True,
            clock=lambda: self.now,
        )


def test_scoring_timeout_preserves_whole_authorized_pool_and_charges_once():
    scenario = Scenario()
    result = scenario.prepare()
    assert result.score_status == "rank_fallback"
    assert result.fallback_reason == "deadline"
    assert [c.chunk_id for c in result.candidates] == [1, 2]
    assert [c.relevance for c in result.candidates] == [1.0, 0.0]
    assert [c.text for c in result.candidates] == [c.content for c in scenario.chunks]
    assert scenario.budget.remaining_ms() == 11999
    assert scenario.budget.can_publish()
    assert scenario.budget.pairs_used == {"acquisition": 0, "final": 1}
    assert result.new_pairs == len(scenario.calls) == 1
    assert 0 < scenario.calls[0][1] <= 3
    assert scenario.budget.scoring_remaining_ms("final") == 0
    assert not scenario.budget.start_pair(phase="final")


@pytest.mark.parametrize("stop", ["initial", "global", "cancel", "revalidation"])
def test_preparation_or_global_closure_never_publishes_fallback(stop):
    scenario = Scenario(stop=stop)
    with pytest.raises((SourcePreparationLimited, ContextLimited)):
        scenario.prepare()
    assert len(scenario.calls) == (0 if stop == "initial" else 1)


@pytest.mark.parametrize("stop, remaining", [("revoke", []), ("revise", [2])])
def test_timeout_fallback_rechecks_current_authority_and_revision(stop, remaining):
    scenario = Scenario(stop=stop)
    result = scenario.prepare()
    assert [c.chunk_id for c in result.candidates] == remaining
    assert result.score_status == "rank_fallback"


@pytest.mark.asyncio
@pytest.mark.django_db(transaction=True)
async def test_scoring_timeout_support_reaches_real_handoff_and_synthesis(monkeypatch, close_direct_worker):
    from channels.db import database_sync_to_async
    from apps.chat.services import rag_turn
    from apps.chat.services.rag_evidence import build_selected_evidence_packet
    from apps.chat.services.rag_selection import select_evidence
    from apps.chat.services.rag_selection_types import SelectionLimits, SelectionProfile
    from apps.chat.services.rag_synthesis import synthesize_from_evidence
    from apps.chat.tests.chat_message_test_support import _FakeLLMInterface
    from apps.chat.tests.persisted_direct_test_support import persisted_direct_case
    from apps.documents.models import TextChunk
    from apps.documents.services.source_loading import source_query_rows
    from lib.llm.evidence_guard import current_protection
    from lib.llm.types.conversation import Conversation
    from lib.llm.types.messages import ToolMessage, UserMessage
    from lib.llm.types.response import LLMResponse

    scenario = Scenario()
    declared = [dict(doc_id="scoring", chunk_id=i, title="Scoring source", text=c.content,
        citation=f"[doc:scoring chunk:{i}]") for i, c in enumerate(scenario.chunks, 1)]
    case = await persisted_direct_case(declared)
    convo = Conversation(system="system", messages=[UserMessage(content="result?"),
        ToolMessage(tool_name="vector_search", for_whom="assistant",
            content="retrieval preview", arguments={}, result_dict={})])
    consumer = case.consumer(convo, (1,))
    monkeypatch.setenv("RAG_EVIDENCE_TEXT_MODE", "source")
    monkeypatch.setenv("OPENAI_CONTEXT_LIMIT", "16000")
    monkeypatch.setattr(rag_turn, "TurnBudget", lambda limits: scenario.budget)
    delivered = []
    async with rag_turn.preservation_turn(consumer, max_func_calls=3) as runtime:
        assert runtime.budget is scenario.budget
        original = runtime._frozen_retrieval_scope.representation
        scenario.authorization = runtime.authorization
        def load():
            return tuple(source_query_rows(TextChunk.objects.filter(
                pk__in=[c.pk for c in case.chunks.values()]).order_by("chunk_number")))
        scenario.chunks = await database_sync_to_async(load)()
        prepared = await database_sync_to_async(scenario.prepare)()
        assert prepared.score_status == "rank_fallback" and prepared.fallback_reason == "deadline"
        assert scenario.budget.scoring_remaining_ms("final") == 0
        assert scenario.budget.remaining_ms() == 11999
        selected = select_evidence(prepared.candidates,
            profile=SelectionProfile("test", 1, 0, "v1"), limits=SelectionLimits(2, 2, 3500),
            score_status=prepared.score_status)
        packet = build_selected_evidence_packet(selected, query="result?", search_scope="selected")
        packet.source_mode, packet.source_authorization, packet.selection = True, runtime.authorization, selected
        answer = "42 mK only below 1 Pa; written consent required. " + " ".join(
            f"[doc:{c.doc_id} chunk:{c.pk}]" for c in scenario.chunks)
        llm = _FakeLLMInterface([LLMResponse(text=answer, tool_call=None,
            stop_reason="end_turn", input_usage=1, output_usage=1)])
        original_get = llm.get_message
        async def get_message(*args, **kwargs):
            protection = current_protection()
            assert protection is not None
            for chunk in scenario.chunks:
                assert chunk.content in protection.payload
                assert f"[doc:{chunk.doc_id} chunk:{chunk.pk}]" in protection.payload
            assert runtime._frozen_retrieval_scope.representation is original
            delivered.append(protection.synthesis_lease)
            return await original_get(*args, **kwargs)
        llm.get_message = get_message
        result = await synthesize_from_evidence(llm, convo, packet)
        assert "42 mK only below 1 Pa" in result[-1].content
        assert result[-1]._validated_answer is not None
        assert len(llm.calls) == len(delivered) == 1 and delivered[0].can_publish()
        assert scenario.budget.pairs_used["final"] == len(scenario.calls) == 1
    from asyncio import CancelledError
    with pytest.raises(CancelledError, match="^finished$"):
        delivered[0].can_publish()
