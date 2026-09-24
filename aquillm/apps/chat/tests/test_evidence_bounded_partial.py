"""Bounded answers require observed facts and explicit reviewed answer portions."""

import json
from copy import deepcopy

import pytest

from apps.chat.evals.evidence_quality_eval import text_digest


def partial_case():
    source = {
        "source_id": "s",
        "revision": "r",
        "text": "Avery owns review.",
        "authorized_at_answer": True,
    }
    span = {
        "source_id": "s",
        "revision": "r",
        "fingerprint": text_digest(source["text"]),
        "start": 0,
        "end": len(source["text"]),
        "text": source["text"],
    }
    packet = {
        "doc_id": "doc",
        "chunk_id": 1,
        "text": source["text"],
        "citation": "[doc:doc chunk:1]",
    }
    payload = {"content": json.dumps({"result": [packet]})}
    answer = (
        "Avery owns review. [doc:doc chunk:1] Limit reached; other owners are unknown."
    )
    row = {
        "answer": answer,
        "delivered": [span],
        "upstream": [span],
        "citations": [{"source_id": "s", "revision": "r"}],
        "sdk_payloads": [payload],
        "source_bindings": [
            {"document_id": "doc", "chunk_id": 1, "source_id": "s", "revision": "r"}
        ],
        "events": [
            {"event": "retrieval_sources", "rows": [packet]},
            {"event": "sdk_start", "payload": payload},
        ],
    }
    fact = {
        "fact_id": "f",
        "span": span,
        "acquisition_event": 0,
        "answer_start": 0,
        "answer_end": 36,
        "citation": "[doc:doc chunk:1]",
        "entailed": True,
        "qualifications_correct": True,
    }
    checks = {
        k: True
        for k in (
            "preserves_usable_support",
            "explicit_limit",
            "explicit_missing_aspects",
            "no_fabrication",
            "qualifications_correct",
            "citations_entailed",
            "losses_explained",
            "finalization_reserve_preserved",
        )
    }
    checks.update(
        available_facts=[fact],
        retained_fact_ids=["f"],
        usable_fact_count=1,
        retained_fact_count=1,
        losses=[],
        limit_answer_span={"start": 36, "end": len(answer), "text": answer[36:]},
    )
    return {"sources": [source]}, row, checks


@pytest.mark.parametrize(
    "mutation",
    ["empty", "no_delivery", "count", "stale", "no_citation", "fake_acquisition"],
)
def test_partial_rejects_unobserved_assertions(mutation):
    from apps.chat.evals.evidence_bounded_partial import bounded_partial

    case, row, checks = partial_case()
    assert bounded_partial(case, row, checks)
    if mutation == "empty":
        row["answer"] = ""
    if mutation == "no_delivery":
        row["delivered"] = []
    if mutation == "count":
        checks["usable_fact_count"] = 2
    if mutation == "stale":
        checks["available_facts"][0]["span"]["revision"] = "old"
    if mutation == "no_citation":
        row["citations"] = []
    if mutation == "fake_acquisition":
        checks["available_facts"][0]["acquisition_event"] = 1
    assert not bounded_partial(case, row, checks)


def test_zero_support_needs_actual_empty_acquisition_not_a_reason_string():
    from apps.chat.evals.evidence_bounded_partial import bounded_partial

    case, row, checks = partial_case()
    row.update(delivered=[], upstream=[], citations=[])
    row["answer"] = (
        "Limit reached; no sources were available and all owners remain unknown."
    )
    checks["limit_answer_span"] = {
        "start": 0,
        "end": len(row["answer"]),
        "text": row["answer"],
    }
    row["events"][0]["rows"] = []
    row["sdk_payloads"] = [{"content": "No sources available"}]
    row["events"][1]["payload"] = row["sdk_payloads"][0]
    checks.update(
        available_facts=[],
        retained_fact_ids=[],
        usable_fact_count=0,
        retained_fact_count=0,
        empty_reason="No usable sources",
        empty_event_refs=[0],
    )
    assert bounded_partial(case, row, checks)
    invalid = deepcopy(checks)
    invalid["empty_event_refs"] = []
    assert not bounded_partial(case, row, invalid)


def test_observed_partial_survives_bound_operational_rescore():
    from apps.chat.evals.evidence_operational import assess
    from apps.chat.evals.evidence_quality_eval import digest
    from apps.chat.evals.evidence_review_subject import review_subject
    from apps.chat.tests.test_evidence_operational import observation
    from apps.chat.tests.test_evidence_review_subject import reviewed_case

    case, partial, checks = partial_case()
    _, row, review = reviewed_case()
    ledger = observation()
    ledger["events"].insert(1, {"event": "rerank_http", "pairs": 90})
    row.update(ledger)
    partial["events"] = row["events"] + partial["events"]
    row.update(partial)
    checks["available_facts"][0]["acquisition_event"] = 4
    row.update(case_id="pressure", profile="pilot")
    row["snapshot"]["source"] = digest(case["sources"])
    workload = {
        "workload_id": "pressure",
        "source_ids": ["s"],
        "question": "Owners?",
        "turns": [],
        "required_support": [],
        "ideal_support": [],
    }
    review.update(
        answer_sha256=text_digest(row["answer"]),
        subject=review_subject(row),
        bounded_partial=checks,
    )
    result = assess(workload, case, row, review)
    assert result["operational"]["limit_pass"]
    assert assess(workload, case, result, result["human_review"])["operational"][
        "limit_pass"
    ]
    result["sdk_payloads"] = []
    assert not assess(workload, case, result, review)["operational"]["limit_pass"]
