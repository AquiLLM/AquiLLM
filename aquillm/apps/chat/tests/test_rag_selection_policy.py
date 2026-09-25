"""Intent and score-normalization contracts for adaptive evidence selection."""

import math

import pytest

from apps.chat.services.rag_selection_policy import (
    choose_selection_profile,
    normalize_relevance_scores,
)


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Who led the trial?", "focused"),
        ("When was the trial completed?", "focused"),
        ("Where did it take place?", "focused"),
        ("How many people enrolled?", "focused"),
        ("Compare the two trials.", "breadth"),
        ("Who reported the difference across studies?", "breadth"),
        ("Summarize the literature review.", "breadth"),
        ("What does the result imply?", "balanced"),
        ("Who led it, and what did it conclude?", "balanced"),
        ("What is comparison with the baseline?", "balanced"),
        ("How does retry logic work?", "balanced"),
        ("Who wrote the Comparison of Trials paper?", "focused"),
    ],
)
def test_profile_uses_question_cues_with_word_boundaries(question, expected):
    assert choose_selection_profile(question).name == expected


def test_explicit_single_document_target_is_focused():
    assert (
        choose_selection_profile("Explain the method", single_document=True).name
        == "focused"
    )
    assert (
        choose_selection_profile("Compare findings", single_document=True).name
        == "breadth"
    )


def test_quoted_document_title_does_not_turn_factual_question_into_breadth():
    assert choose_selection_profile('Who wrote "Compare Trials"?').name == "focused"


def test_retry_reuses_the_same_resolved_question_profile():
    resolved_question = "Compare results across studies"
    first = choose_selection_profile(resolved_question)
    choose_selection_profile("Who enrolled?")
    assert choose_selection_profile(resolved_question) == first


def test_profiles_keep_versioned_coefficients():
    profiles = [
        choose_selection_profile(q) for q in ("Who?", "Explain this", "Compare papers")
    ]
    assert [(p.name, p.relevance_weight, p.gap_allowance) for p in profiles] == [
        ("focused", 0.95, 0.05),
        ("balanced", 0.90, 0.10),
        ("breadth", 0.80, 0.15),
    ]
    assert all(
        profile.version and profile.version == profiles[0].version
        for profile in profiles
    )


def test_minmax_preserves_gaps_and_constant_tolerance():
    assert normalize_relevance_scores((10.0, 20.0, 30.0)) == (0.0, 0.5, 1.0)
    assert normalize_relevance_scores(()) == ()
    assert normalize_relevance_scores((7.0,)) == (0.5,)
    assert normalize_relevance_scores((1e9, 1e9 + 0.5)) == (0.5, 0.5)
    assert normalize_relevance_scores((1e9, 1e9 + 2.0)) == (0.0, 1.0)


@pytest.mark.parametrize(
    "scores", [(True,), (math.nan,), (math.inf,), (1.0, -math.inf)]
)
def test_minmax_rejects_invalid_scores(scores):
    with pytest.raises(ValueError):
        normalize_relevance_scores(scores)
