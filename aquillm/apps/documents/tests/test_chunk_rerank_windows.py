"""Exact source coverage regressions for bounded reranker preparation."""

import pytest

from apps.documents.services.chunk_rerank_windows import prepare_source_windows
from lib.retrieval import SourceEvidence, TurnBudget, TurnLimits


def count_pair(query, text):
    return len(query) + len(text) + 8


@pytest.mark.parametrize(
    "text",
    [
        "Background. " * 120 + "Treatment did not improve survival at 5 mg/kg.",
        "x " * 480 + "Treatment did not improve survival. " * 5,
        "ΔE = mc²; 5 μg/kg 😀. " * 90,
    ],
)
def test_windows_cover_exact_tail_and_preserve_qualifications(text):
    source = SourceEvidence(1, "paper-a", 0, "revision-a", text)
    plan = prepare_source_windows(
        "Did survival improve?", source, pair_counter=count_pair
    )
    assert plan.preparation_coverage == "complete"
    assert plan.windows[0].start == 0
    assert plan.windows[-1].end == len(text)
    assert all(w.text == text[w.start : w.end] for w in plan.windows)
    assert all(count_pair(plan.query, w.text) <= 1024 for w in plan.windows)
    assert all(a.end >= b.start for a, b in zip(plan.windows, plan.windows[1:]))
    if "did not" in text:
        assert any("did not improve survival" in w.text for w in plan.windows)


def test_full_pair_first_and_query_never_shortened():
    source = SourceEvidence(1, "paper-a", 0, "r", "tail")
    assert (
        prepare_source_windows("q", source, pair_counter=count_pair).windows[0].text
        == "tail"
    )
    plan = prepare_source_windows("q" * 1024, source, pair_counter=count_pair)
    assert plan.query == "q" * 1024
    assert not plan.windows
    assert plan.preparation_coverage == "partial"


def test_unknown_counter_retains_full_source_and_unknown_coverage():
    source = SourceEvidence(1, "paper-a", 0, "r", "Δ " * 1500)
    plan = prepare_source_windows("q", source, pair_counter=lambda q, d: None)
    assert plan.source == source
    assert plan.windows[-1].end == len(source.text)
    assert plan.preparation_coverage == "unknown"


def test_repeated_counting_is_charged_and_stops_on_budget():
    budget = TurnBudget(TurnLimits(tokenized_codepoints=100))
    plan = prepare_source_windows(
        "q",
        SourceEvidence(1, "d", 0, "r", "x" * 200),
        pair_counter=count_pair,
        budget=budget,
    )
    assert plan.preparation_coverage == "partial"
    assert not plan.windows


@pytest.mark.parametrize("final_ms, coordinator_ms", [(100, 3000), (3000, 100)])
def test_pool_preparation_stops_before_next_source_at_nested_deadline(
    final_ms, coordinator_ms
):
    from types import SimpleNamespace
    from uuid import UUID

    from apps.documents.services.chunk_rerank_window_adapter import (
        WindowSelectionScorer,
    )

    now, counted = [0.0], []
    budget = TurnBudget(TurnLimits(final_scoring_ms=final_ms), clock=lambda: now[0])

    def slow_count(query, document):
        counted.append(document)
        now[0] += 0.2
        return len(query) + len(document)

    rows = tuple(
        SimpleNamespace(pk=i, doc_id=UUID(int=i), chunk_number=0, content=f"source {i}")
        for i in range(1, 4)
    )
    scorer = WindowSelectionScorer(
        None,
        budget=budget,
        pair_counter=slow_count,
        clock=lambda: now[0],
        deadline=coordinator_ms / 1000,
    )
    result = scorer.score_windows("q", rows)
    assert counted == ["source 1"]
    assert now[0] == 0.2
    assert result.status == "unavailable"
    assert budget.pairs_used == {"acquisition": 0, "final": 0}
    assert budget.can_publish()  # Final fallback must not close the retrieval ledger.


def test_expired_window_preparation_does_not_start_another_counter_operation():
    now, counted = [0.0], []
    source = SourceEvidence(1, "doc", 0, "r", "x" * 3000)

    def slow_count(query, document):
        counted.append(len(document))
        now[0] += 0.2
        return len(query) + len(document)

    plan = prepare_source_windows(
        "q", source, pair_counter=slow_count, deadline=0.1, clock=lambda: now[0]
    )
    assert counted == [3000]
    assert plan.source is source
    assert plan.preparation_coverage == "partial"
    assert plan.reason == "preparation_deadline"
    assert not plan.windows


def test_rendering_that_exhausts_deadline_cannot_start_tokenization():
    now, tokenized = [0.0], []

    class SlowRender:
        def input_codepoints(self, query, document):
            now[0] += 0.2
            return len(query) + len(document)

        def __call__(self, query, document):
            tokenized.append(document)
            return 1

    plan = prepare_source_windows(
        "q",
        SourceEvidence(1, "doc", 0, "r", "tail"),
        pair_counter=SlowRender(),
        deadline=0.1,
        clock=lambda: now[0],
    )
    assert not tokenized
    assert plan.preparation_coverage == "partial"
    assert plan.reason == "preparation_deadline"


def test_reuse_preparation_uses_coordinator_deadline_for_injected_scorer(monkeypatch):
    from types import SimpleNamespace
    from uuid import UUID

    from apps.chat.services.rag_selection_scoring import prepare_selection_candidates
    from apps.documents.services.chunk_rerank_results import fingerprint_text
    from apps.documents.services.chunk_rerank_window_adapter import (
        WindowSelectionScorer,
    )

    now, counted = [0.0], []
    budget = TurnBudget(TurnLimits(), clock=lambda: now[0])
    rows = tuple(
        SimpleNamespace(pk=i, doc_id=UUID(int=i), chunk_number=0, content=f"source {i}")
        for i in range(1, 4)
    )
    acquisition = WindowSelectionScorer(
        SimpleNamespace(score_pair=lambda pair, timeout: (1.0, pair)),
        budget=budget,
        pair_counter=count_pair,
        scorer_identity="deadline-fixture",
        clock=lambda: now[0],
        deadline=10,
    ).score_windows("q", rows, phase="acquisition")

    def slow_count(query, document):
        counted.append(document)
        now[0] += 0.2
        return count_pair(query, document)

    final = WindowSelectionScorer(
        None,
        budget=budget,
        pair_counter=slow_count,
        scorer_identity="deadline-fixture",
        clock=lambda: now[0],
        deadline=10,
    )
    public = tuple({"chunk_id": row.pk, "citation": str(row.pk)} for row in rows)
    hydrated = tuple(
        SimpleNamespace(
            chunk=row,
            row=view,
            excerpt=row.content,
            source_fingerprint=fingerprint_text(row.content),
        )
        for row, view in zip(rows, public)
    )
    monkeypatch.setattr(
        "apps.chat.services.rag_selection_scoring.hydrate_pool_rows",
        lambda *args, **kwargs: hydrated,
    )
    pool = SimpleNamespace(
        rows=public, source_score_sets=(acquisition,), fused_scores=()
    )
    result = prepare_selection_candidates(
        pool=pool,
        primary_query="q",
        authorization=object(),
        deadline=0.1,
        allow_new_scores=False,
        scorer=final,
        clock=lambda: now[0],
        turn_budget=budget,
    )
    assert counted == ["source 1"]
    assert result.score_status == "rank_fallback"
    assert now[0] == 0.2


@pytest.mark.parametrize("delayed_method", ["reserve_text", "can_publish"])
def test_ledger_wait_cannot_start_counter_after_deadline(monkeypatch, delayed_method):
    now, operations = [0.0], []
    budget = TurnBudget(TurnLimits(), clock=lambda: now[0])
    original = getattr(budget, delayed_method)
    calls = 0

    def delayed(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = original(*args, **kwargs)
        if delayed_method == "reserve_text" or calls == 2:
            now[0] += 0.2
        return result

    class Counter:
        def input_codepoints(self, query, document):
            operations.append("render")
            return len(query) + len(document)

        def __call__(self, query, document):
            operations.append("tokenize")
            return 1

    monkeypatch.setattr(budget, delayed_method, delayed)
    plan = prepare_source_windows(
        "q",
        SourceEvidence(1, "doc", 0, "r", "tail"),
        pair_counter=Counter(),
        budget=budget,
        deadline=0.1,
        clock=lambda: now[0],
    )
    assert operations == (["render"] if delayed_method == "reserve_text" else [])
    assert budget.text_used["tokenized"] == (
        5 if delayed_method == "reserve_text" else 0
    )
    assert plan.reason == "preparation_deadline"
    assert plan.preparation_coverage == "partial"
    assert not plan.windows


def test_plan_admission_rechecks_deadline_after_ledger_wait(monkeypatch):
    from apps.documents.services.chunk_rerank_window_adapter import (
        WindowSelectionScorer,
    )

    now = [0.0]
    budget = TurnBudget(TurnLimits(), clock=lambda: now[0])
    original = budget.can_publish

    def delayed():
        result = original()
        now[0] += 0.2
        return result

    class UnreadSource:
        @property
        def pk(self):
            pytest.fail("source preparation started after the nested deadline")

    monkeypatch.setattr(budget, "can_publish", delayed)
    scorer = WindowSelectionScorer(
        None, budget=budget, deadline=0.1, clock=lambda: now[0]
    )
    with pytest.raises(ValueError, match="allowance exhausted"):
        scorer.plan("q", UnreadSource())
