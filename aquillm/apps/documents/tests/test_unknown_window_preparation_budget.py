"""Unknown pair fit cannot consume the budget for usable evidence handoff."""

from threading import Event
from types import SimpleNamespace
from uuid import UUID

import pytest

from apps.documents.services import chunk_rerank_pair_capability as capability
from apps.documents.services.chunk_rerank_window_adapter import (
    WindowSelectionScorer,
    unknown_pair_count,
)
from apps.documents.services.chunk_rerank_window_scores import (
    aggregate_window_scores,
    score_window_plan,
)
from apps.documents.services.chunk_rerank_windows import prepare_source_windows
from lib.llm.evidence_guard import estimate_request_tokens
from lib.retrieval import SourceEvidence, TurnBudget, TurnLimits


class Scorer:
    def score_pair(self, pair, timeout_seconds):
        pytest.fail("unknown inputs reached inference")


def source(number=1):
    return SourceEvidence(number, "synthetic", 0, "revision", "x " * 961)


def rows():
    return tuple(
        SimpleNamespace(
            pk=i, doc_id=UUID(int=i), chunk_number=0, content=source(i).text
        )
        for i in range(1, 46)
    )


def test_unknown_multi_source_planning_leaves_real_request_estimation_headroom():
    budget = TurnBudget(TurnLimits())
    query, counted = "q" * 200, []

    def unknown(query, document):
        counted.append(len(query) + len(document))
        return None

    try:
        sources = tuple(source(i) for i in range(1, 46))
        plans = tuple(
            prepare_source_windows(query, item, pair_counter=unknown, budget=budget)
            for item in sources
        )
        assert all(plan.source is item for plan, item in zip(plans, sources))
        assert all(plan.preparation_coverage == "unknown" for plan in plans)
        assert all(plan.windows == () for plan in plans)
        assert len(counted) == 45
        assert budget.text_used["tokenized"] == 45 * (200 + 1922)
        assert (
            estimate_request_tokens(
                {"messages": ["handoff" * 1000]}, turn_budget=budget
            )
            > 0
        )
        assert budget.pairs_used == {"acquisition": 0, "final": 0}
        assert budget.can_publish()
    finally:
        budget.close("unknown_planner_test_finished")


@pytest.mark.parametrize("missing", ["provider", "counter"])
def test_unavailable_adapter_keeps_rank_order_without_planning_work(missing):
    budget = TurnBudget(TurnLimits())
    candidates, counted = rows(), []

    def count(query, document):
        counted.append(document)
        return len(query) + len(document) + 8

    scorer = WindowSelectionScorer(
        None if missing == "provider" else Scorer(),
        budget=budget,
        pair_counter=count if missing == "provider" else unknown_pair_count,
    )
    try:
        result = scorer.score_windows("q" * 200, candidates, phase="acquisition")
        assert result.candidate_order == tuple(range(1, 46))
        assert result.status == "unavailable" and result.scoring_kind == "rank_only"
        assert result.scores == ()
        assert budget.text_used["tokenized"] == 0
        assert counted == []
        assert budget.pairs_used == {"acquisition": 0, "final": 0}
        assert (
            estimate_request_tokens(
                {"messages": ["handoff" * 1000]}, turn_budget=budget
            )
            > 0
        )
    finally:
        budget.close("unknown_adapter_test_finished")


@pytest.mark.parametrize(
    "loss", ["expired", "revoked", "expires_later", "revokes_later"]
)
def test_real_pair_proof_loss_stops_at_first_unknown_count(monkeypatch, loss):
    now, tokenizations, revoked = [0.0], [], Event()
    monkeypatch.setattr(capability, "monotonic", lambda: now[0])

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return messages[0]["content"] + messages[1]["content"] + "template"

        def encode(self, rendered, **kwargs):
            tokenizations.append(len(rendered))
            if loss == "expires_later":
                now[0] = 2.0
            elif loss == "revokes_later":
                revoked.set()
            return list(range(len(rendered)))

    counter = capability.VerifiedPairCounter(
        Tokenizer(), "template", "proof", "tokenizer", "template", 1.0, revoked
    )
    if loss == "expired":
        now[0] = 2.0
    elif loss == "revoked":
        revoked.set()
    budget = TurnBudget(TurnLimits())
    item = source()
    try:
        plan = prepare_source_windows("q", item, pair_counter=counter, budget=budget)
        assert plan.source is item
        assert plan.preparation_coverage == "unknown"
        assert plan.reason == "pair_counter_unknown"
        assert plan.windows == ()
        assert len(tokenizations) == int(loss.endswith("later"))
        # One full pair render, and only a base render after proof loss mid-plan.
        assert budget.text_used["tokenized"] == 1931 + (
            9 if loss.endswith("later") else 0
        )
        scored = score_window_plan(plan, scorer=Scorer(), budget=budget)
        assert scored.attempted_pairs == 0
        assert (
            aggregate_window_scores(scored).prepared_input_scoring_coverage == "unknown"
        )
    finally:
        budget.close("proof_loss_test_finished")


def test_lost_counter_discards_unscorable_prior_windows_but_keeps_original_source():
    budget, counts = TurnBudget(TurnLimits()), []
    item = source()

    def counter(query, document):
        counts.append(len(document))
        return len(query) + len(document) + 8 if len(counts) <= 25 else None

    try:
        plan = prepare_source_windows("q", item, pair_counter=counter, budget=budget)
        assert plan.source is item and plan.source.text == "x " * 961
        assert plan.preparation_coverage == "unknown"
        assert len(counts) == 26
        assert plan.windows == ()
        assert (
            score_window_plan(plan, scorer=Scorer(), budget=budget).attempted_pairs == 0
        )
    finally:
        budget.close("mid_plan_loss_test_finished")


def test_unknown_caption_counter_never_estimates_image_or_text_windows():
    budget = TurnBudget(TurnLimits())
    chunk = rows()[0]
    chunk.modality = "image"
    scorer = WindowSelectionScorer(
        Scorer(), budget=budget, pair_counter=unknown_pair_count
    )
    try:
        plan = scorer.plan("q", chunk)
        assert plan.source.text == chunk.content
        assert plan.preparation_coverage == "unknown" and plan.windows == ()
        assert scorer.score_windows("q", (chunk,)).status == "unavailable"
        assert budget.pairs_used == {"acquisition": 0, "final": 0}
        assert budget.text_used["tokenized"] == 1923
    finally:
        budget.close("unknown_image_test_finished")
