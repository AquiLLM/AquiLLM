"""Untrusted hints may reorder the existing frontier, never force a source."""
from dataclasses import replace
from hashlib import sha256

import pytest

from apps.chat.services.rag_selection import select_evidence
from apps.chat.services.rag_selection_types import SelectionCandidate, SelectionLimits, SelectionProfile


def candidate(pk, score, *, document=None, cost=10):
    text = f"Independent finding {pk} describes its measured condition."
    return SelectionCandidate(pk, document or f"doc-{pk}", pk, text, score, pk,
        sha256(text.encode()).hexdigest(), {}, token_cost=cost, raw_score=score,
        scorer_fingerprint="same-query-scorer", scoring_kind="pointwise")


def choose(pool, hints=(), *, passages=3, tokens=40, per_doc=2, status="model"):
    return select_evidence(pool, profile=SelectionProfile("test", .9, .1, "v1"),
        limits=SelectionLimits(passages, per_doc, tokens), score_status=status,
        support_hints=hints)


def test_general_hint_orders_only_inside_original_relevance_frontier():
    pool = tuple(candidate(i, score) for i, score in enumerate((.95, .90, .89, .02), 1))
    result = choose(pool, ((pool[2], ("method-condition",)), (pool[3], ("wrong-head",))), passages=2)
    assert result.candidates == (pool[2], pool[0])
    assert all(row is pool[row.chunk_id - 1] for row in result.candidates)
    assert [row.raw_score for row in result.candidates] == [.89, .95]


def test_same_aspect_conflicting_and_secondary_references_remain_distinct():
    pool = tuple(candidate(i, score) for i, score in enumerate((.95, .94, .93), 1))
    hints = ((pool[1], (("effect", "source-a", 0, 8),)),
             (pool[2], (("effect", "secondary-condition", 0, 8),)))
    assert choose(pool, hints, passages=2).candidates == (pool[1], pool[2])


def test_increasing_passage_capacity_preserves_hint_selection_prefix():
    pool = tuple(candidate(i, score) for i, score in enumerate((.95, .92, .91, .90, .01), 1))
    hints = ((pool[2], ("earlier",)), (pool[3], ("condition",)), (pool[4], ("irrelevant",)))
    selected = [choose(pool, hints, passages=n, tokens=60).candidates for n in (2, 3, 4, 5)]
    assert [row.chunk_id for row in selected[0]] == [3, 4]
    assert all(later[:len(earlier)] == earlier for earlier, later in zip(selected, selected[1:]))


def test_hint_cannot_exceed_token_or_document_feasibility():
    pool = (candidate(1, .95, document="same"), candidate(2, .94, document="same"),
            candidate(3, .93, cost=40))
    result = choose(pool, ((pool[1], ("condition",)), (pool[2], ("expensive",))),
                    tokens=19, per_doc=1)
    assert result.candidates == (pool[1],)
    assert result.estimated_tokens == 10


@pytest.mark.parametrize("failure", ["partial", "rank", "missing", "scorer", "constant", "near_constant"])
def test_unusable_scores_disable_hinting_without_changing_ordinary_selection(failure):
    pool = tuple(candidate(i, score) for i, score in enumerate((.95, .94, .93), 1))
    status = "model"
    if failure in ("partial", "rank"):
        status = "partial_model_ranked" if failure == "partial" else "rank_fallback"
    elif failure == "missing":
        pool = (*pool[:2], replace(pool[2], raw_score=None))
    elif failure == "scorer":
        pool = (*pool[:2], replace(pool[2], scorer_fingerprint="different-query"))
    else:
        pool = tuple(replace(row, raw_score=.01 + (i * 1e-12 if failure == "near_constant" else 0))
                     for i, row in enumerate(pool))
    ordinary = choose(pool, status=status)
    assert choose(pool, ((pool[2], ("condition",)),), status=status) == ordinary


def test_absent_hints_preserve_existing_selector_and_raw_scale_is_not_confidence():
    pool = tuple(candidate(i, score) for i, score in enumerate((.95, .94, .93), 1))
    ordinary = select_evidence(pool, profile=SelectionProfile("test", .9, .1, "v1"),
        limits=SelectionLimits(3, 2, 40), score_status="model")
    assert choose(pool) == ordinary
    lower = tuple(replace(row, raw_score=row.raw_score / 100) for row in pool)
    for current in (pool, lower):
        assert choose(current, ((current[2], ("condition",)),), passages=1).candidates == (current[2],)


def test_foreign_hint_cannot_insert_or_replace_a_current_candidate():
    pool = (candidate(1, .9), candidate(2, .85))
    with pytest.raises(ValueError, match="hint"):
        choose(pool, ((replace(pool[1]), ("condition",)),))
