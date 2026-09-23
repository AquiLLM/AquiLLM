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
