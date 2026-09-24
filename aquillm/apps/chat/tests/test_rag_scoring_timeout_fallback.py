"""A scoring-only timeout preserves exact support without extending its allowance."""

from types import SimpleNamespace

import pytest

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
async def test_scoring_timeout_support_reaches_real_handoff_and_synthesis(monkeypatch):
    from apps.chat.services import rag_source_hydration
    from apps.chat.services.rag_evidence import build_selected_evidence_packet
    from apps.chat.services.rag_selection import select_evidence
    from apps.chat.services.rag_selection_types import SelectionLimits, SelectionProfile
    from apps.chat.services.rag_synthesis import synthesize_from_evidence
    from apps.documents.services.source_loading import (
        SourceRuntime,
        source_runtime_scope,
    )
    from lib.llm.evidence_guard import current_protection
    from lib.llm.types.conversation import Conversation
    from lib.llm.types.messages import AssistantMessage, ToolMessage, UserMessage

    scenario = Scenario()
    prepared = scenario.prepare()
    selected = select_evidence(
        prepared.candidates,
        profile=SelectionProfile("test", 1, 0, "v1"),
        limits=SelectionLimits(2, 2, 3500),
        score_status=prepared.score_status,
    )
    packet = build_selected_evidence_packet(
        selected, query="result?", search_scope="selected"
    )
    packet.source_mode, packet.source_authorization, packet.selection = (
        True,
        scenario.authorization,
        selected,
    )
    monkeypatch.setenv("OPENAI_CONTEXT_LIMIT", "16000")
    # Only the external body fetch is replaced; bounded DB/current policy and
    # revision checks, packet rebuild, handoff and synthesis sealing stay real.
    monkeypatch.setattr(
        rag_source_hydration, "source_query_rows", lambda *_: scenario.chunks
    )
    delivered = []

    async def complete(request, *_args, **_kwargs):
        protection = current_protection()
        assert protection is not None and protection.payload == request[-1].content
        for chunk in scenario.chunks:
            assert chunk.content in protection.payload
            assert f"[doc:{chunk.doc_id} chunk:{chunk.pk}]" in protection.payload
        delivered.append(protection.synthesis_lease)
        return request + [
            AssistantMessage(
                content="42 mK only below 1 Pa; written consent required.",
                stop_reason="end_turn",
            )
        ], "changed"

    with source_runtime_scope(SourceRuntime(scenario.budget, scenario.authorization)):
        result = await synthesize_from_evidence(
            SimpleNamespace(complete=complete),
            Conversation(
                system="system",
                messages=[
                    UserMessage(content="result?"),
                    ToolMessage(
                        tool_name="vector_search",
                        for_whom="assistant",
                        content="retrieval preview",
                        arguments={},
                        result_dict={},
                    ),
                ],
            ),
            packet,
        )
    assert "42 mK only below 1 Pa" in result[-1].content
    assert len(delivered) == 1 and delivered[0].can_publish()
    assert scenario.budget.pairs_used["final"] == len(scenario.calls) == 1
