"""Comparison coverage preserves actual candidate scores and native resource caps."""

import pytest

from apps.chat.services.rag_selection import select_evidence
from apps.chat.services.rag_selection_types import (
    SelectionCandidate,
    SelectionLimits,
    SelectionProfile,
)


def candidates():
    return tuple(
        SelectionCandidate(
            chunk_id=index,
            doc_id=document,
            chunk_number=index,
            text=f"Source passage {index} for {document}.",
            relevance=score,
            fused_rank=index,
            source_fingerprint=str(index) * 64,
            row={},
            token_cost=10,
            raw_score=score,
            scorer_fingerprint="real-scorer",
            scoring_kind="pointwise",
        )
        for index, document, score in (
            (1, "earlier", 0.2), (2, "earlier", 0.1),
            (3, "later", 0.3), (4, "later", 0.15),
            (5, "background", 0.9),
        )
    )


def test_complete_comparison_groups_retain_original_scores_and_objects():
    pool = candidates()
    selected = select_evidence(
        pool,
        profile=SelectionProfile("comparison-test", 0.7, 0.1, "v1"),
        limits=SelectionLimits(5, 2, 50),
        score_status="model",
        coverage_groups=(pool[:2], pool[2:4]),
    )
    assert selected.candidates == pool
    assert all(actual is original for actual, original in zip(selected.candidates, pool, strict=True))
    assert selected.estimated_tokens == 50
    assert [candidate.raw_score for candidate in selected.candidates] == [0.2, 0.1, 0.3, 0.15, 0.9]


@pytest.mark.parametrize("limits", [SelectionLimits(3, 2, 50), SelectionLimits(5, 1, 50), SelectionLimits(5, 2, 19)])
def test_infeasible_two_sided_coverage_abstains_without_widening_caps(limits):
    pool = candidates()
    kwargs = dict(profile=SelectionProfile("comparison-test", 0.7, 0.1, "v1"),
                  limits=limits, score_status="model")
    ordinary = select_evidence(pool, **kwargs)
    selected = select_evidence(pool, coverage_groups=(pool[:2], pool[2:4]), **kwargs)
    assert selected == ordinary


def test_one_sided_or_rank_fallback_coverage_abstains():
    pool = candidates()
    kwargs = dict(profile=SelectionProfile("comparison-test", 0.7, 0.1, "v1"),
                  limits=SelectionLimits(5, 2, 50), score_status="rank_only")
    ordinary = select_evidence(pool, **kwargs)
    assert select_evidence(pool, coverage_groups=(pool[:2], pool[2:4]), **kwargs) == ordinary
    kwargs["score_status"] = "model"
    assert select_evidence(pool, coverage_groups=(pool[:2],), **kwargs) == select_evidence(pool, **kwargs)


def test_foreign_caller_candidate_cannot_enter_coverage():
    pool = candidates()
    foreign = candidates()[0]
    kwargs = dict(profile=SelectionProfile("comparison-test", 0.7, 0.1, "v1"),
                  limits=SelectionLimits(5, 2, 50), score_status="model")
    with pytest.raises(ValueError, match="coverage"):
        select_evidence(pool, coverage_groups=((foreign, pool[1]), pool[2:4]), **kwargs)
