"""Selection of paired, source-bounded sections through the adaptive selector."""

from dataclasses import replace

import pytest

from apps.chat.services.rag_selection import select_evidence
from apps.chat.services.rag_selection_policy import choose_selection_profile
from apps.chat.services.rag_selection_types import SelectionCandidate, SelectionLimits
from lib.retrieval.evidence import (
    PreparedEvidence,
    SourceEvidence,
    SourceSpan,
    fingerprint_prepared_evidence,
    fingerprint_source,
)

QUESTION = (
    "Which filtration media changed between the 2019 and 2022 river station protocols?"
)


def passage(pk, owner, number, body, relevance, cost=20):
    fingerprint = fingerprint_source(body)
    source = SourceEvidence(pk, owner, number, fingerprint, body)
    spans = (SourceSpan(pk, fingerprint, 0, len(body), body),)
    evidence = PreparedEvidence(
        source, spans, fingerprint_prepared_evidence(source, spans), cost, "complete"
    )
    return SelectionCandidate(
        pk, owner, number, body, relevance, pk, fingerprint, {}, evidence, cost
    )


def comparison_pool():
    # Both manuals use the retained medium's trade name. Only the independent
    # field register gives its standard designation.
    overlap = (
        "The columns were rinsed with distilled water before each run. "
        "A witnessed field blank preceded each change of cartridge. "
    )

    def section(pk, owner, first, last):
        return (
            passage(
                pk,
                owner,
                3,
                f"- Filtration media: The protocol specifies {first}. " + overlap,
                0.23,
            ),
            passage(
                pk + 1,
                owner,
                4,
                overlap
                + f"Use {last} at the station.\n- Field blanks: Next procedure.",
                0.21,
            ),
        )

    return (
        passage(1, "2019 manual", 0, "Overview of the 2019 river protocols.", 0.99),
        passage(2, "2022 manual", 0, "River station protocol overview for 2022.", 0.98),
        passage(
            3, "2019 manual", 8, "General history of the catchment and station.", 0.97
        ),
        passage(
            4, "2022 manual", 8, "Index to reporting forms at the study site.", 0.96
        ),
        passage(
            5,
            "lab register",
            0,
            "The field register identifies Amberlite as resin A-32.",
            0.92,
        ),
        *section(11, "2019 manual", "Amberlite", "glass fiber"),
        *section(21, "2022 manual", "Amberlite", "cellulose acetate"),
    )


def choose(
    pool, *, question=QUESTION, passages=5, budget=100, per_doc=3, mode="adaptive"
):
    return select_evidence(
        pool,
        profile=choose_selection_profile(question),
        limits=SelectionLimits(passages, per_doc, budget),
        score_status="rank_fallback",
        mode=mode,
        question=question,
    )


def test_comparison_preserves_both_printed_sections_and_the_independent_register():
    pool = comparison_pool()
    result = choose(pool)
    assert [c.chunk_id for c in result.candidates] == [11, 12, 21, 22, 5]
    assert (
        result.estimated_tokens == sum(c.token_cost for c in result.candidates) == 100
    )
    assert all(any(original is c for original in pool) for c in result.candidates)


@pytest.mark.parametrize(
    "change",
    [
        "missing_tail",
        "mismatched_overlap",
        "omitted_text",
        "over_budget",
        "owner_cap",
        "single_owner",
        "unrelated_question",
        "legacy",
    ],
)
def test_ineligible_pair_preserves_the_ordinary_selection(change):
    pool = list(comparison_pool())
    options = {}
    if change == "missing_tail":
        pool.pop()
    elif change == "mismatched_overlap":
        pool[-1] = passage(
            22, "2022 manual", 4, "Unrelated continuation.\n- Field blanks: Next.", 0.21
        )
    elif change == "omitted_text":
        row = pool[-1]
        spans = (row.prepared_evidence.spans[0],)
        part = replace(spans[0], start=80, text=row.text[80:])
        reduced = (part,)
        evidence = PreparedEvidence(
            row.prepared_evidence.source,
            reduced,
            fingerprint_prepared_evidence(row.prepared_evidence.source, reduced),
            5,
            "partial",
        )
        pool[-1] = replace(
            row, prepared_evidence=evidence, text=part.text, token_cost=5
        )
    elif change == "over_budget":
        options["budget"] = 59
    elif change == "owner_cap":
        options["per_doc"] = 1
    elif change == "single_owner":
        pool = pool[:-2]
    elif change == "unrelated_question":
        options["question"] = (
            "Which filtration media does the 2019 river protocol require?"
        )
    else:
        options["mode"] = "legacy"
    result = choose(pool, **options)
    ordinary = choose(pool, **(options | {"question": "Explain the protocol."}))
    assert result.candidates == ordinary.candidates
    assert result.estimated_tokens == ordinary.estimated_tokens


def test_nonoverlapping_rows_require_contiguous_native_offsets():
    pool = list(comparison_pool())
    for head_index in (5, 7):
        head = pool[head_index]
        tail = pool[head_index + 1]
        continuation = tail.text[tail.text.index("Use ") :]
        pool[head_index] = replace(head, source_start=500)
        pool[head_index + 1] = replace(
            passage(
                tail.chunk_id,
                tail.doc_id,
                tail.chunk_number,
                continuation,
                tail.relevance,
            ),
            source_start=500 + len(head.text),
        )
    assert [c.chunk_id for c in choose(pool).candidates] == [11, 12, 21, 22, 5]
    pool[-1] = replace(pool[-1], source_start=pool[-1].source_start + 1)
    assert [c.chunk_id for c in choose(pool).candidates] == [1, 2, 3, 4, 5]


def test_three_candidate_owners_with_the_same_heading_do_not_establish_a_unique_pair():
    pool = (
        *comparison_pool(),
        passage(
            30,
            "lab register",
            9,
            "- Filtration media: ISO examples include membranes.\n- Labels: Next.",
            0.2,
        ),
    )
    assert [c.chunk_id for c in choose(pool).candidates] == [1, 2, 3, 4, 5]


def test_heading_from_another_prepared_source_cannot_narrow_the_question():
    pool = list(comparison_pool())
    row = pool[-2]
    other = passage(50, "unscoped document", 3, row.text, 0.3)
    pool[-2] = replace(row, prepared_evidence=other.prepared_evidence)
    assert [c.chunk_id for c in choose(pool).candidates] == [1, 2, 3, 4, 5]
