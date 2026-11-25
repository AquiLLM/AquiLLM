"""Follow-up matching can order whole scored candidates without creating evidence."""

from dataclasses import replace

import pytest

from apps.chat.services.rag_planner_context import AcquisitionAttempt
from apps.chat.services.rag_selection import select_evidence
from apps.chat.services.rag_selection_policy import normalize_relevance_scores
from apps.chat.services.rag_selection_types import (
    SelectionCandidate,
    SelectionLimits,
    SelectionProfile,
)
from lib.retrieval.evidence import (
    PreparedEvidence,
    SourceEvidence,
    SourceSpan,
    fingerprint_prepared_evidence,
    fingerprint_source,
)
from lib.retrieval.turn_budget import TurnBudget, TurnLimits


def trial_pool():
    bodies = [
        "AX-4 storage guidance for broad collections. "
        "Keep ordinary pieces in cabinets.",
        "Storage guidance for the loading room and for paper.",
        "Storage guidance for temporary storage and shelving.",
        "The AX-4 amber seal belongs on the borosilicate vessel only after curing.",
        "Storage guidance for stacked shelving units and shelf labels.",
        "Storage guidance for assembly and site work.",
    ]
    raw = (0.80, 0.72, 0.70, 0.69, 0.60, 0.30)
    pool = []
    for i, (body, score, rel) in enumerate(
        zip(bodies, raw, normalize_relevance_scores(raw)), 1
    ):
        digest = fingerprint_source(body)
        source = SourceEvidence(i, "guide", i, digest, body)
        span = SourceSpan(i, digest, 0, len(body), body)
        prepared = PreparedEvidence(
            source,
            (span,),
            fingerprint_prepared_evidence(source, (span,)),
            20,
            "complete",
        )
        pool.append(
            SelectionCandidate(
                i,
                "guide",
                i,
                body,
                rel,
                i,
                digest,
                {},
                prepared,
                20,
                score,
                "same-model",
                "pointwise",
            )
        )
    return tuple(pool)


def select(pool, followup, *, budget=None, status="model", mode="adaptive"):
    from apps.chat.services.rag_selection_coordinator import select_prepared_evidence

    return select_prepared_evidence(
        pool,
        profile=SelectionProfile("test", 1, 0, "v1"),
        limits=SelectionLimits(2, 2, 40),
        score_status=status,
        mode=mode,
        question="How is AX-4 storage handled?",
        followup_query=followup,
        turn_budget=budget or TurnBudget(TurnLimits()),
    )


def test_named_followup_ranks_exact_original_row_with_its_original_score():
    pool = trial_pool()
    result = select(pool, "AX-4 amber borosilicate curing")
    assert result.candidates[0] is pool[0]
    assert result.candidates[1] is pool[3]
    assert result.candidates[1].raw_score == 0.69
    assert result.profile.version.endswith("+followup-bm25-rrf-v1")


def test_pure_selector_input_reproduces_fusion_without_a_live_query_or_ledger(
    monkeypatch,
):
    from apps.chat.services import rag_selection_coordinator

    pool = trial_pool()
    observed = []

    def capture(candidates, **kwargs):
        observed.append((candidates, kwargs))
        return select_evidence(candidates, **kwargs)

    monkeypatch.setattr(rag_selection_coordinator, "select_evidence", capture)
    result = select(pool, "AX-4 amber borosilicate curing")
    candidates, options = observed[0]
    assert not {"followup_query", "turn_budget"} & options.keys()
    assert candidates[0].relevance > candidates[3].relevance > candidates[1].relevance
    assert [c.raw_score for c in candidates] == [c.raw_score for c in pool]
    assert [c.chunk_id for c in select_evidence(candidates, **options).candidates] == [
        c.chunk_id for c in result.candidates
    ]
    assert all(any(c is original for original in pool) for c in result.candidates)


@pytest.mark.parametrize(
    "followup",
    [
        "unrelated storage advice",
        "one unique amber",
        "unmentioned jargon",
        "AX-40 amber borosilicate",
        "except AX-4 amber borosilicate",
        "amber borosilicate curing",
    ],
)
def test_incidental_or_missing_followup_cannot_displace_evidence(followup):
    pool = trial_pool()
    assert select(pool, followup).candidates == pool[:2]
    assert select(pool, followup).profile.version == "v1"


def test_exhausted_budget_partial_scores_flat_scores_and_legacy_keep_original_order():
    pool = trial_pool()
    query = "AX-4 amber borosilicate curing"
    ledger = TurnBudget(TurnLimits(tokenized_codepoints=1))
    closed = TurnBudget(TurnLimits())
    closed.close("cancelled")
    assert select(pool, query, budget=ledger).candidates == pool[:2]
    assert select(pool, query, budget=closed).candidates == pool[:2]
    assert select(pool, query, status="partial_model_ranked").candidates == pool[:2]
    assert select(pool, query, mode="legacy").candidates == pool[:2]
    flat = tuple(replace(c, raw_score=0.6, relevance=0.5) for c in pool)
    assert select(flat, query).candidates == flat[:2]


def test_only_completed_distinct_vector_followup_with_new_sources_can_rank():
    from apps.chat.services.rag_followup_ranking import completed_followup

    first = AcquisitionAttempt("vector", "Original?", None, None, "completed", 4, 4)
    follow = AcquisitionAttempt(
        "vector", "Amber borosilicate", None, None, "completed", 6, 2
    )
    assert completed_followup((first, follow), "Original?") == follow.query
    for change in (
        {"outcome": "no_progress"},
        {"outcome": "timeout"},
        {"kind": "read"},
        {"new_source_count": 0},
        {"query_truncated": True},
        {"query": "original?"},
    ):
        assert not completed_followup((first, replace(follow, **change)), "Original?")


def test_empty_single_candidate_and_unpayable_scan_leave_synthesis_capacity():
    from lib.llm.evidence_guard import _encoded_estimate_input, estimate_request_tokens

    pool = trial_pool()
    query = "AX-4 amber borosilicate curing"
    assert not select((), query).candidates
    assert select(pool[:1], query).candidates == pool[:1]
    request = {"messages": [{"role": "tool", "content": pool[0].text + pool[1].text}]}
    required = len(_encoded_estimate_input(request))
    ledger = TurnBudget(TurnLimits(tokenized_codepoints=required))
    ordinary = select(pool, query, budget=ledger)
    assert ordinary.candidates == pool[:2]
    assert ledger.text_used["tokenized"] == 0
    assert estimate_request_tokens(
        request, turn_budget=ledger
    ) == estimate_request_tokens(request)
    assert ledger.text_used["tokenized"] == required


def test_payable_lexical_scan_cannot_starve_a_payable_synthesis_request():
    from lib.llm.evidence_guard import _encoded_estimate_input, estimate_request_tokens

    pool = trial_pool()
    query = "AX-4 amber borosilicate curing"
    question = "How is AX-4 storage handled?"
    scan = len(question) + len(query) + sum(len(c.text) for c in pool)
    request = {"messages": [{"role": "tool", "content": pool[0].text + pool[1].text}]}
    required = len(_encoded_estimate_input(request))
    assert 0 < required < scan
    ledger = TurnBudget(TurnLimits(tokenized_codepoints=scan + required - 1))
    selected = select(pool, query, budget=ledger)
    assert selected.profile.version.endswith("+followup-bm25-rrf-v1")
    assert estimate_request_tokens(
        request, turn_budget=ledger
    ) == estimate_request_tokens(request)
    assert ledger.text_used["tokenized"] == required


def test_repeated_lexical_scans_share_a_single_bounded_allowance():
    pool = trial_pool()
    query = "AX-4 amber borosilicate curing"
    scan = (
        len("How is AX-4 storage handled?")
        + len(query)
        + sum(len(c.text) for c in pool)
    )
    ledger = TurnBudget(TurnLimits(materialized_codepoints=scan))
    assert select(pool, query, budget=ledger).profile.version.endswith(
        "+followup-bm25-rrf-v1"
    )
    assert select(pool, query, budget=ledger).profile.version == "v1"
    assert ledger.text_used == {"materialized": 0, "tokenized": 0}


def test_lexical_scan_reservations_are_atomic_and_end_at_closure():
    from concurrent.futures import ThreadPoolExecutor

    ledger = TurnBudget(TurnLimits(materialized_codepoints=30))
    assert ledger.reserve_text(10, kind="materialized")
    with ThreadPoolExecutor(max_workers=4) as workers:
        results = list(workers.map(ledger.reserve_lexical_scan, (10,) * 4))
    assert sorted(results) == [0, 0, 10, 20]
    assert ledger.text_used == {"materialized": 10, "tokenized": 0}
    ledger = TurnBudget(TurnLimits())
    ledger.close("cancelled")
    assert ledger.reserve_lexical_scan(1) == 0
    for invalid in (True, 0, -1, 1.5):
        with pytest.raises(ValueError):
            ledger.reserve_lexical_scan(invalid)


@pytest.mark.parametrize(
    "followup",
    [
        "Atlas-Core-Max storage capacity",
        "Cloud-Atlas-Core and Atlas-Core-Max",
        "Atlas-Core.Max and Atlas-Core-Max",
        "Atlas-Core+Plus and Atlas-Core-Max",
        "atlas-core and Atlas-Core-Max",
        "‘Atlas-Core-Max’ maintenance",
    ],
)
def test_longer_technical_identifier_cannot_supply_a_sibling_anchor(followup):
    from apps.chat.services.rag_followup_ranking import retains_question_anchors

    assert not retains_question_anchors(
        "Compare Atlas-Core and Atlas-Core-Max.", followup
    )


def test_distinct_technical_ids_accept_typographic_hyphens_and_possessives():
    from apps.chat.services.rag_followup_ranking import retains_question_anchors

    question = "Compare Atlas‐Core’s capacity with Atlas‐Core‐Max’s."
    assert not retains_question_anchors(question, "Atlas-Core-Max’s capacity")
    assert retains_question_anchors(
        question, "Atlas-Core’s capacity vs. Atlas-Core-Max"
    )
