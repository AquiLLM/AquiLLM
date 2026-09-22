"""Pure evidence selection behavior and hard-budget boundaries."""

import math
from dataclasses import FrozenInstanceError

import pytest

from apps.chat.services.rag_selection import select_evidence
from apps.chat.services.rag_selection_policy import choose_selection_profile
from apps.chat.services.rag_selection_similarity import snippet_redundancy
from apps.chat.services.rag_selection_types import (
    SelectionCandidate,
    SelectionLimits,
    SelectionProfile,
)


def candidate(pk, doc, text, relevance, *, rank=None, number=None, row=None):
    row = (
        row
        if row is not None
        else {
            "chunk_id": pk,
            "doc_id": doc,
            "text": text,
            "citation": f"[doc:{doc} chunk:{number if number is not None else pk}]",
        }
    )
    return SelectionCandidate(
        pk,
        doc,
        number if number is not None else pk,
        text,
        relevance,
        rank if rank is not None else pk,
        str(pk),
        row,
    )


def chosen(
    pool,
    *,
    question="Explain this",
    max_passages=3,
    per_doc=3,
    budget=1000,
    score_status="model",
):
    return select_evidence(
        pool,
        profile=choose_selection_profile(question),
        limits=SelectionLimits(max_passages, per_doc, budget),
        score_status=score_status,
    )


def test_complementary_second_passage_beats_weak_new_source():
    pool = (
        candidate(1, "a", "The trial measured 12 percent improvement.", 0.95),
        candidate(2, "a", "Calibration used a held-out validation cohort.", 0.92),
        candidate(3, "b", "The introduction discusses related terminology.", 0.25),
    )
    assert [item.chunk_id for item in chosen(pool, max_passages=2).candidates] == [1, 2]


def test_breadth_near_tie_prefers_distinct_source_when_same_document_repeats():
    repeated = "The trial found a twelve percent improvement in outcomes."
    pool = (
        candidate(1, "a", repeated, 0.96),
        candidate(2, "a", repeated, 0.95),
        candidate(3, "b", "The second trial required prior treatment failure.", 0.94),
    )
    assert [
        c.chunk_id
        for c in chosen(pool, question="Compare the trials", max_passages=2).candidates
    ] == [1, 3]


def test_distinct_second_passage_remains_eligible_in_breadth_mode():
    pool = (
        candidate(1, "a", "The trial found a twelve percent improvement.", 0.96),
        candidate(2, "a", "Calibration used a held-out validation cohort.", 0.95),
        candidate(3, "b", "The second trial mentioned a different condition.", 0.80),
    )
    assert [
        c.chunk_id
        for c in chosen(pool, question="Compare the trials", max_passages=2).candidates
    ] == [1, 2]


def test_same_verified_identity_is_deduplicated_without_mutating_rows():
    row = {
        "chunk_id": 1,
        "doc_id": "a",
        "text": "Original",
        "citation": "[doc:a chunk:1]",
    }
    pool = (
        candidate(1, "a", "Original", 0.8, row=row),
        candidate(1, "a", "Original", 0.9, row=row),
        candidate(2, "a", "Original", 0.7),
    )
    result = chosen(pool)
    assert [c.chunk_id for c in result.candidates] == [1, 2]
    assert result.candidates[0].relevance == 0.9
    assert row == {
        "chunk_id": 1,
        "doc_id": "a",
        "text": "Original",
        "citation": "[doc:a chunk:1]",
    }
    with pytest.raises(FrozenInstanceError):
        result.estimated_tokens = 0


def test_same_text_from_different_documents_is_eligible_for_corroboration():
    pool = (
        candidate(1, "a", "The treatment reduced symptoms by twelve percent.", 0.95),
        candidate(2, "b", "The treatment reduced symptoms by twelve percent.", 0.94),
    )
    assert [c.chunk_id for c in chosen(pool, max_passages=2).candidates] == [1, 2]


def test_redundancy_uses_unicode_three_shingles_and_attenuates_cross_document():
    text = "Café patients showed 12 percent improvement"
    same = candidate(1, "a", text, 0.9)
    repeated = candidate(2, "a", "CAFÉ patients showed 12 percent improvement!", 0.8)
    other = candidate(3, "b", text, 0.7)
    assert snippet_redundancy(same, repeated) == 1.0
    assert snippet_redundancy(same, other) == 0.5
    assert (
        snippet_redundancy(
            candidate(4, "a", "Yes", 0.8), candidate(5, "a", "yes!", 0.7)
        )
        == 1.0
    )
    assert (
        snippet_redundancy(candidate(4, "a", "Yes", 0.8), candidate(5, "a", "no", 0.7))
        == 0.0
    )


@pytest.mark.parametrize(
    "second",
    [
        "The treatment did not reduce symptoms by twelve percent.",
        "The treatment reduced symptoms by 23 percent.",
        "Under severe disease, the treatment reduced symptoms by twelve percent.",
    ],
)
def test_changed_claims_keep_redundancy_below_exact_repeat(second):
    first = candidate(1, "a", "The treatment reduced symptoms by twelve percent.", 0.9)
    assert snippet_redundancy(first, candidate(2, "a", second, 0.8)) < 1.0


def test_exact_fit_oversized_and_zero_budget():
    pool = (
        candidate(1, "a", "A" * 8, 1.0),
        candidate(2, "b", "B" * 9, 0.9),
        candidate(3, "c", "C" * 8, 0.8),
    )
    result = chosen(pool, budget=4)
    assert [c.chunk_id for c in result.candidates] == [1, 3]
    assert result.estimated_tokens == 4
    assert chosen(pool, budget=0).candidates == ()
    assert chosen((), budget=4).estimated_tokens == 0


def test_oversized_candidate_does_not_consume_document_slot():
    pool = (
        candidate(1, "a", "A" * 100, 1.0),
        candidate(2, "a", "Short", 0.8),
        candidate(3, "b", "Other", 0.7),
    )
    assert [c.chunk_id for c in chosen(pool, per_doc=1, budget=4).candidates] == [2, 3]


def test_per_document_ceiling_applies_with_one_document():
    pool = tuple(candidate(i, "a", f"Finding {i}", 1 - i / 100) for i in range(1, 5))
    result = chosen(pool, question="Who conducted this study?", per_doc=2)
    assert len(result.candidates) == 2


def test_compact_and_full_public_rows_do_not_change_selection():
    full = (
        candidate(1, "a", "First finding", 0.9),
        candidate(2, "b", "Other finding", 0.8),
    )
    compact = tuple(
        candidate(
            c.chunk_id,
            c.doc_id,
            c.text,
            c.relevance,
            row={"d": c.doc_id, "x": c.text, "i": c.chunk_number},
        )
        for c in full
    )
    assert [c.chunk_id for c in chosen(full).candidates] == [
        c.chunk_id for c in chosen(compact).candidates
    ]


def test_equal_scores_use_rank_then_stable_identity_not_input_order():
    pool = (
        candidate(3, "z", "Third distinct finding", 0.5, rank=2),
        candidate(2, "b", "Second distinct finding", 0.5, rank=1),
        candidate(1, "a", "First distinct finding", 0.5, rank=1),
    )
    assert [c.chunk_id for c in chosen(pool).candidates] == [1, 2, 3]


def test_rank_only_status_uses_the_same_budget_loop():
    pool = (candidate(1, "a", "Very long " * 20, 1.0), candidate(2, "b", "Short", 0.5))
    result = chosen(pool, budget=2, score_status="rank_only")
    assert [c.chunk_id for c in result.candidates] == [2]
    assert result.score_status == "rank_only"


def test_forty_five_candidate_boundary_and_reject_over_cap():
    pool = tuple(
        candidate(i, str(i), f"Finding {i}", 1 - i / 100) for i in range(1, 46)
    )
    assert len(chosen(pool, max_passages=45).candidates) == 45
    with pytest.raises(ValueError):
        chosen(pool + (candidate(46, "46", "extra", 0.1),), max_passages=45)


@pytest.mark.parametrize(
    "limits",
    [SelectionLimits(-1, 1, 1), SelectionLimits(1, 0, 1), SelectionLimits(1, 1, -1)],
)
def test_invalid_limits_are_rejected(limits):
    with pytest.raises(ValueError):
        select_evidence(
            (),
            profile=choose_selection_profile("Explain"),
            limits=limits,
            score_status="model",
        )


@pytest.mark.parametrize(
    "profile",
    [
        SelectionProfile("custom", math.nan, 0.1, "v1"),
        SelectionProfile("custom", 1.1, 0.1, "v1"),
        SelectionProfile("custom", 0.8, -0.1, "v1"),
    ],
)
def test_invalid_profile_configuration_is_rejected(profile):
    with pytest.raises(ValueError):
        select_evidence(
            (), profile=profile, limits=SelectionLimits(1, 1, 1), score_status="model"
        )
