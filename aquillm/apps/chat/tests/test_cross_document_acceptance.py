"""Cross-document regressions; scripted provider text is not model-quality proof."""

from __future__ import annotations

import pytest

from apps.chat.tests.chat_message_test_support import _FakeLLMInterface
from apps.chat.tests.cross_document_acceptance_support import (
    CASES,
    evaluate_answer,
    run_acceptance_case,
)
from lib.llm.types.response import LLMResponse


def _probe(case, text=None):
    # Fixed text only lets the real completion/citation path finish. Assertions
    # below measure its inputs, not the quality of this hand-authored answer.
    response = LLMResponse(
        text=text or case.reference_answer,
        tool_call={},
        stop_reason="end_turn",
        input_usage=1,
        output_usage=1,
    )
    return _FakeLLMInterface([response] * 4)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.case_id)
async def test_both_paper_facts_survive_fusion_before_final_two_row_limit(case):
    report = await run_acceptance_case(case, _probe(case), retrieval_limit=2)

    assert report["outcome"] == "handled"
    assert report["evidence_coverage_passed"], report["evidence_citations"]
    assert len(report["requests"][0]["tool_payload"]["result"]) == 2
    assert report["serialized_rows_match"]


async def test_provider_receives_only_packet_rows_and_their_exact_citations():
    case = CASES[0]
    report = await run_acceptance_case(case, _probe(case), retrieval_limit=4)

    assert report["outcome"] == "handled"
    assert set(report["evidence_citations"]) == case.required_citations
    assert report["exact_allowlist_passed"]
    assert report["serialized_rows_match"]
    assert "apparatus description" not in report["requests"][0]["tool_content"]


async def test_prior_turn_citations_do_not_expand_current_packet_allowlist():
    case = CASES[0]
    old = {
        "doc_id": "synthetic-old",
        "chunk_id": 999,
        "title": "Old synthetic paper",
        "text": "This old trial is outside the current evidence packet.",
        "citation": "[doc:synthetic-old chunk:999]",
    }
    report = await run_acceptance_case(
        case,
        _probe(case),
        retrieval_limit=4,
        prior_tool_rows=[old],
    )

    assert report["outcome"] == "handled"
    assert report["exact_allowlist_passed"]
    assert old["citation"] not in report["requests"][0]["system_citations"]


async def test_final_completion_repairs_citation_to_prior_evidence_outside_packet():
    case = CASES[0]
    old = {
        "doc_id": "synthetic-old",
        "chunk_id": 999,
        "title": "Old synthetic paper",
        "text": "This old trial is outside the current evidence packet.",
        "citation": "[doc:synthetic-old chunk:999]",
    }
    invalid_answer = (
        case.reference_answer + f"\nThe old trial is relevant {old['citation']}."
    )
    llm = _FakeLLMInterface(
        [
            LLMResponse(
                text=text,
                tool_call={},
                stop_reason="end_turn",
                input_usage=1,
                output_usage=1,
            )
            for text in [invalid_answer, case.reference_answer]
        ]
    )
    report = await run_acceptance_case(
        case, llm, retrieval_limit=4, prior_tool_rows=[old]
    )

    assert report["outcome"] == "handled"
    assert old["citation"] not in report["answer"]
    assert report["provider_call_count"] == 2
    assert report["exact_allowlist_passed"]


async def test_deferred_final_stream_delivers_the_repaired_stored_answer(monkeypatch):
    monkeypatch.setenv("LLM_STREAM_FINAL_ANSWER_ONLY", "1")
    case = CASES[0]
    invalid_citation = "[doc:synthetic-unavailable chunk:999]"
    invalid_answer = case.reference_answer.replace(
        case.papers[0]["citation"],
        invalid_citation,
    )
    llm = _FakeLLMInterface(
        [
            LLMResponse(
                text=text,
                tool_call={},
                stop_reason="end_turn",
                input_usage=1,
                output_usage=1,
            )
            for text in [invalid_answer, case.reference_answer]
        ]
    )

    report = await run_acceptance_case(case, llm, stream_output=True)

    assert report["outcome"] == "handled"
    assert report["provider_call_count"] == 2
    assert invalid_citation not in report["answer"]
    assert report["final_stream_received"] is True
    assert report["final_stream_content_matches_answer"] is True
    assert report["stream_event_count"] == 1
    assert all(call.get("stream_callback") is None for call in llm.calls)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.case_id)
@pytest.mark.parametrize("omitted_index", [0, 1])
async def test_single_paper_ablation_removes_its_evidence_and_citation(
    case, omitted_index
):
    omitted = case.papers[omitted_index]
    retained = case.papers[1 - omitted_index]
    probe_answer = f"The available source remains in context {retained['citation']}."
    report = await run_acceptance_case(
        case,
        _probe(case, probe_answer),
        omitted_doc_id=omitted["doc_id"],
    )

    assert report["outcome"] == "handled"
    assert set(report["evidence_citations"]) == {retained["citation"]}
    assert report["exact_allowlist_passed"]
    assert omitted["citation"] not in report["requests"][0]["tool_content"]
    # The model-quality grader is intentionally not asserted against probe text.


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.case_id)
def test_answer_checker_accepts_reference_and_rejects_missing_second_fact(case):
    assert evaluate_answer(
        case,
        case.reference_answer,
        allowed_citations=case.required_citations,
    )["passed"]
    first_claim_only = case.reference_answer.splitlines()[0]
    report = evaluate_answer(
        case, first_claim_only, allowed_citations=case.required_citations
    )
    assert f"missing-claim:{case.claims[1].name}" in report["errors"]
    assert f"missing-claim:{case.synthesis.name}" in report["errors"]


def test_answer_checker_rejects_swapped_citations_and_source_inventory_only():
    case = CASES[0]
    first, second = [paper["citation"] for paper in case.papers]
    swapped = (
        case.reference_answer.replace(first, "TEMP")
        .replace(second, first)
        .replace("TEMP", second)
    )
    report = evaluate_answer(case, swapped, allowed_citations=case.required_citations)
    assert "wrong-or-missing-citation:dry-capacity" in report["errors"]
    assert "wrong-or-missing-citation:humid-capacity" in report["errors"]
    uncited = case.reference_answer.replace(first, "").replace(second, "")
    report = evaluate_answer(
        case,
        uncited + f"\nSources:\n{first}\n{second}",
        allowed_citations=case.required_citations,
    )
    assert "wrong-or-missing-citation:cross-paper-difference" in report["errors"]


def test_answer_checker_accepts_enhances_paraphrase_only_with_local_citation():
    case = CASES[1]
    first_citation = case.papers[0]["citation"]
    answer_lines = case.reference_answer.splitlines()
    answer_lines[0] = (
        "Activating the coating enhances response by 12 percent in the "
        f"controlled trial {first_citation}."
    )
    report = evaluate_answer(
        case,
        "\n".join(answer_lines),
        allowed_citations=case.required_citations,
    )
    assert report["passed"], report["errors"]
    answer_lines[0] = answer_lines[0].replace(first_citation, "")
    uncited = evaluate_answer(
        case,
        "\n".join(answer_lines),
        allowed_citations=case.required_citations,
    )
    assert "wrong-or-missing-citation:controlled-increase" in uncited["errors"]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.case_id)
@pytest.mark.parametrize("omitted_index", [0, 1])
def test_answer_checker_ablation_requires_insufficiency_without_inventing_facts(
    case,
    omitted_index,
):
    omitted = case.papers[omitted_index]
    retained = case.papers[1 - omitted_index]
    retained_claim = case.reference_answer.splitlines()[1 - omitted_index]
    report = evaluate_answer(
        case,
        retained_claim
        + "\nThe other paper is missing; the comparison cannot be determined.",
        allowed_citations={retained["citation"]},
        omitted_doc_id=omitted["doc_id"],
    )
    assert report["passed"], report["errors"]
    hallucinated = evaluate_answer(
        case,
        case.reference_answer,
        allowed_citations={retained["citation"]},
        omitted_doc_id=omitted["doc_id"],
    )
    assert (
        f"unsupported-claim:{case.claims[omitted_index].name}" in hallucinated["errors"]
    )
    assert "citations-outside-evidence" in hallucinated["errors"]
    assert "missing-insufficiency-statement" in hallucinated["errors"]
