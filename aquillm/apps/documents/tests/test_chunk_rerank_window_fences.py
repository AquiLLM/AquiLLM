"""Actual provider calls share slots, retry limits, and late-publication fences."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event, Lock
from time import monotonic
from types import SimpleNamespace
from unittest.mock import patch

from apps.documents.services.chunk_rerank_selection_provider import LocalSelectionScorer
from lib.retrieval import TurnBudget, TurnLimits


def provider():
    return LocalSelectionScorer(
        endpoint="http://test/score",
        shape="score_single_text_pair",
        model_name="m",
        revision="r",
        char_limit=2000,
        pair_limit=1024,
        reserve=256,
        timeout=3,
        deadline=monotonic() + 3,
    )


def test_independent_providers_share_six_actual_transports_and_reject_late_results():
    budget = TurnBudget(TurnLimits())
    gate, release, six = Barrier(9), Event(), Event()
    lock = Lock()
    calls = []

    def post(*args, **kwargs):
        with lock:
            calls.append(kwargs["json"]["text_2"])
            if len(calls) == 6:
                six.set()
        assert release.wait(2)
        return SimpleNamespace(status_code=200, json=lambda: {"score": 1.0})

    def score(index):
        gate.wait(timeout=2)
        return provider().score_budgeted_pair(
            ("q", str(index)), 2, budget=budget, phase="acquisition"
        )

    with patch(
        "apps.documents.services.chunk_rerank_selection_provider.requests.post",
        side_effect=post,
    ):
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(score, i) for i in range(8)]
            gate.wait(timeout=2)
            assert six.wait(2)
            budget.close("cancelled")
            release.set()
            assert all(f.result() is None for f in futures)
    assert len(calls) == 6
    assert budget.pairs_used["acquisition"] == 6


def test_retry_cannot_exceed_shared_actual_pair_allowance():
    budget = TurnBudget(TurnLimits(acquisition_pairs=1))
    response = SimpleNamespace(status_code=400)
    with patch(
        "apps.documents.services.chunk_rerank_selection_provider.requests.post",
        return_value=response,
    ):
        assert (
            provider().score_budgeted_pair(
                ("q", "long document"), 1, budget=budget, phase="acquisition"
            )
            is None
        )
    assert budget.pairs_used["acquisition"] == 1


def test_shadow_inference_requires_explicit_opt_in_and_uses_existing_allowance(
    monkeypatch,
):
    from uuid import UUID

    from apps.documents.services.chunk_rerank_window_acquisition import (
        dispatch_windowed,
    )
    from apps.documents.services.chunk_rerank_window_adapter import (
        WindowSelectionScorer,
    )

    monkeypatch.setenv("RAG_RERANK_TEXT_MODE", "shadow")
    row = SimpleNamespace(pk=1, doc_id=UUID(int=1), chunk_number=0, content="x" * 2000)
    budget = TurnBudget(TurnLimits(acquisition_pairs=1))
    scorer = WindowSelectionScorer(
        SimpleNamespace(score_pair=lambda p, t: (1.0, p)),
        budget=budget,
        pair_counter=lambda q, d: len(q) + len(d),
    )
    assert dispatch_windowed("q", (row,), 1, budget, scorer, shadow=True) is None
    assert budget.pairs_used["acquisition"] == 0
    monkeypatch.setenv("RAG_RERANK_SHADOW_SCORING_ENABLED", "1")
    assert dispatch_windowed("q", (row,), 1, budget, scorer, shadow=True) is None
    assert budget.pairs_used["acquisition"] == 1
    assert dispatch_windowed("q", (row,), 1, None, shadow=True) is None


def test_timeout_after_one_window_keeps_partial_record_without_complete_aggregate():
    from apps.documents.services.chunk_rerank_window_scores import (
        aggregate_window_scores,
        score_window_plan,
    )
    from apps.documents.services.chunk_rerank_windows import prepare_source_windows
    from lib.retrieval import SourceEvidence

    class Stops:
        calls = 0

        def score_pair(self, pair, timeout_seconds):
            self.calls += 1
            if self.calls == 2:
                raise TimeoutError
            return (1.0, pair)

    plan = prepare_source_windows(
        "q",
        SourceEvidence(1, "d", 0, "r", "x" * 2000),
        pair_counter=lambda q, d: len(q) + len(d),
    )
    budget = TurnBudget(TurnLimits())
    result = score_window_plan(plan, scorer=Stops(), budget=budget)
    assert len(result.successful) == 1
    assert budget.pairs_used["acquisition"] == 2
    assert aggregate_window_scores(result).prepared_input_scoring_coverage == "partial"


def test_unequal_window_counts_use_max_without_candidate_count_normalization():
    from uuid import UUID

    from apps.documents.services.chunk_rerank_window_adapter import (
        WindowSelectionScorer,
    )

    rows = tuple(
        SimpleNamespace(pk=i, doc_id=UUID(int=i), chunk_number=0, content=text)
        for i, text in ((1, "short"), (2, "x" * 1800 + " decisive tail"))
    )
    scorer = WindowSelectionScorer(
        SimpleNamespace(
            score_pair=lambda p, t: (9.0 if "decisive tail" in p[1] else 1.0, p)
        ),
        budget=TurnBudget(TurnLimits()),
        pair_counter=lambda q, d: len(q) + len(d),
    )
    result = scorer.score_windows("q", rows, phase="acquisition")
    assert [s.value for s in result.scores] == [1.0, 9.0]
    assert len(result.scores[0].window_coverage.required_window_ids) == 1
    assert len(result.scores[1].window_coverage.required_window_ids) > 1
