"""Final scoring admits whole source plans under the existing pair allowance."""

from types import SimpleNamespace

import pytest

from apps.chat.services.rag_selection import select_evidence
from apps.chat.services.rag_selection_scoring import prepare_selection_candidates
from apps.chat.services.rag_selection_types import SelectionLimits, SelectionProfile
from apps.chat.tests.test_rag_selection_scoring import (
    PointwiseScorer,
    Policy,
    _authorization,
    _chunk,
    _pool,
    _score_set,
)
from apps.documents.services.chunk_rerank_window_adapter import WindowSelectionScorer
from lib.evidence_observation import observe
from lib.retrieval import TurnBudget, TurnLimits


class Scenario:
    def __init__(self, *, count=45, long_at=43, final_pairs=45, failure=None):
        self.now = 0.0
        self.budget = TurnBudget(TurnLimits(final_pairs=final_pairs), clock=self.clock)
        self.policy = Policy()
        self.chunks = tuple(
            _chunk(i, "long " * 280 if i == long_at else f"source {i}")
            for i in range(1, count + 1)
        )
        self.failure, self.calls, self.events = failure, [], []
        self.graph_id = count - 1
        self.scorer = WindowSelectionScorer(
            SimpleNamespace(scorer_fingerprint="one-model", score_pair=self.score),
            budget=self.budget,
            pair_counter=lambda q, d: len(q) + len(d) + 8,
            deadline=3.0,
            clock=self.clock,
        )

    def clock(self):
        return self.now

    def score(self, pair, timeout):
        self.calls.append(pair)
        if self.failure == "provider":
            return None
        if self.failure == "deadline":
            self.now = 3.001
        if self.failure == "revise":
            self.chunks[self.graph_id - 1].content = "changed input"
        if self.failure == "revoke":
            self.policy.allowed = ()
        if self.failure == "race" and len(self.calls) == 1:
            assert self.budget.reserve_pairs(1, phase="final")
        return (10.0 if pair[1] == f"source {self.graph_id}" else 0.0), pair

    def prepare(self, *, score_sets=(), allow_new=True):
        with observe(lambda event, data: self.events.append((event, data))):
            return prepare_selection_candidates(
                pool=_pool(self.chunks, score_sets), primary_query="q",
                authorization=_authorization(self.policy), deadline=3.0,
                allow_new_scores=allow_new, scorer=self.scorer,
                chunk_loader=lambda *_: self.chunks, turn_budget=self.budget,
                source_mode=True, clock=self.clock,
            )


def test_45_sources_with_46_windows_select_complete_model_subset():
    case = Scenario()
    plans = tuple(case.scorer.plan("q", chunk) for chunk in case.chunks)
    assert sum(len(plan.windows) for plan in plans) == 46
    result = case.prepare()
    assert result.score_status == "model" and result.fallback_reason is None
    assert [item.chunk_id for item in result.candidates] == list(range(1, 45))
    assert result.new_pairs == len(case.calls) == case.budget.pairs_used["final"] == 45
    assert case.budget.limits.final_pairs == 45
    assert case.budget.limits.final_scoring_ms == 3000
    assert case.budget.pairs_used["acquisition"] == 0
    assert all(item.scoring_kind == "pointwise" for item in result.candidates)
    assert all(item.raw_score is not None for item in result.candidates)
    assert all(("q", window.text) in case.calls for window in plans[42].windows)
    selection = select_evidence(
        result.candidates, profile=SelectionProfile("test", 1, 0, "v1"),
        limits=SelectionLimits(12, 12, 7000), score_status=result.score_status,
    )
    assert selection.candidates[0].chunk_id == case.graph_id
    (admission,) = [
        row for event, row in case.events if event == "selection_window_admission"
    ]
    assert admission["candidate_count"] == 45 and admission["admitted_count"] == 44
    assert admission["planned_new_pairs_upper_bound"] == 46
    assert admission["admitted_new_pairs_upper_bound"] == 45


def test_oversized_plan_does_not_block_later_source_that_fits():
    case = Scenario(count=3, long_at=2, final_pairs=2)
    result = case.prepare()
    assert result.score_status == "model"
    assert [item.chunk_id for item in result.candidates] == [1, 3]
    assert case.calls == [("q", "source 1"), ("q", "source 3")]
    assert result.new_pairs == case.budget.pairs_used["final"] == 2


@pytest.mark.parametrize("failure", ["provider", "deadline"])
def test_failed_admitted_scoring_preserves_original_pool_for_fallback(failure):
    case = Scenario(failure=failure)
    result = case.prepare()
    assert result.score_status == "rank_fallback"
    assert [item.chunk_id for item in result.candidates] == list(range(1, 46))
    assert all(item.raw_score is None for item in result.candidates)
    assert case.budget.pairs_used["final"] <= 45


@pytest.mark.parametrize("failure", ["revise", "revoke"])
def test_admitted_model_scores_revalidate_current_source_and_permission(failure):
    case = Scenario(failure=failure)
    result = case.prepare()
    assert case.graph_id not in {item.chunk_id for item in result.candidates}
    if failure == "revoke":
        assert not result.candidates
    else:
        assert result.score_status == "model"
        assert len(result.candidates) == 43


def test_exact_complete_reused_score_consumes_no_new_pair_allowance():
    case = Scenario(count=2, long_at=1, final_pairs=1)
    reused = case.scorer.score_windows("q", case.chunks[:1], phase="acquisition")
    assert reused.status == "complete" and case.budget.pairs_used["acquisition"] == 2
    case.calls.clear()
    result = case.prepare(score_sets=(reused,))
    assert result.score_status == "model" and len(result.candidates) == 2
    assert result.reused_pairs == result.new_pairs == 1
    assert case.calls == [("q", "source 2")]


def test_unknown_window_coverage_does_not_admit_a_model_subset():
    case = Scenario(count=3, long_at=2, final_pairs=2)
    case.scorer.pair_counter = lambda q, d: None if d.startswith("long") else len(d) + 9
    result = case.prepare()
    assert result.score_status == "rank_fallback" and len(result.candidates) == 3
    assert case.calls == []


def test_disabled_new_scoring_does_not_prune_original_pool_to_reused_scores():
    case = Scenario(count=3, long_at=2, final_pairs=2)
    reused = case.scorer.score_windows("q", case.chunks[:1], phase="acquisition")
    case.calls.clear()
    result = case.prepare(score_sets=(reused,), allow_new=False)
    assert result.score_status == "rank_fallback" and len(result.candidates) == 3
    assert case.calls == []


def test_admission_subtracts_already_used_final_pairs():
    case = Scenario(count=3, long_at=2, final_pairs=3)
    assert case.budget.reserve_pairs(1, phase="final")
    result = case.prepare()
    assert result.score_status == "model"
    assert [item.chunk_id for item in result.candidates] == [1, 3]
    assert result.new_pairs == 2 and case.budget.pairs_used["final"] == 3


def test_pair_reservation_race_keeps_transport_cap_and_original_fallback(monkeypatch):
    monkeypatch.setenv("APP_RERANK_SCORE_CONCURRENCY", "1")
    case = Scenario(count=3, long_at=2, final_pairs=2, failure="race")
    result = case.prepare()
    assert result.score_status == "rank_fallback" and len(result.candidates) == 3
    assert result.new_pairs == len(case.calls) == 1
    assert case.budget.pairs_used["final"] == 2


def test_failed_admitted_batch_cannot_publish_only_its_reused_rows():
    case = Scenario(count=5, long_at=2, final_pairs=2)
    reused = case.scorer.score_windows("q", case.chunks[:1], phase="acquisition")
    case.failure = "provider"
    result = case.prepare(score_sets=(reused,))
    assert result.score_status == "rank_fallback" and len(result.candidates) == 5
    assert all(item.raw_score is None for item in result.candidates)


def test_zero_remaining_pairs_preserves_original_ranked_pool():
    case = Scenario(count=3, long_at=2, final_pairs=0)
    result = case.prepare()
    assert result.score_status == "rank_fallback" and len(result.candidates) == 3
    assert result.new_pairs == 0 and case.calls == []


@pytest.mark.parametrize("final_pairs", [2, 3])
def test_listwise_provider_keeps_whole_pool_and_rejects_subset_score_reuse(final_pairs):
    budget = TurnBudget(TurnLimits(final_pairs=final_pairs), clock=lambda: 0.0)
    chunks = tuple(_chunk(i, f"source {i}") for i in range(1, 4))
    calls = []

    class ListwiseScorer(PointwiseScorer):
        scoring_kind = "listwise"

        def score_pair(self, *_):
            pytest.fail("pool-dependent provider must receive the whole ordered pool")

        def score_pool(self, pairs, timeout):
            calls.append(pairs)
            if not budget.start_pairs(len(pairs), phase="final"):
                return None
            try:
                return tuple((float(i), pair) for i, pair in enumerate(pairs))
            finally:
                budget.finish_pairs(len(pairs))

    result = prepare_selection_candidates(
        pool=_pool(chunks, (_score_set("q", chunks[:1], (9.0,), kind="listwise"),)),
        primary_query="q", authorization=_authorization(Policy()), deadline=3.0,
        allow_new_scores=True, scorer=ListwiseScorer(),
        chunk_loader=lambda *_: chunks, turn_budget=budget, clock=lambda: 0.0,
    )
    assert calls == [tuple(("q", chunk.content) for chunk in chunks)]
    assert len(result.candidates) == 3 and result.reused_pairs == 0
    assert result.score_status == ("model" if final_pairs == 3 else "rank_fallback")
    assert budget.pairs_used["final"] == (3 if final_pairs == 3 else 0)
