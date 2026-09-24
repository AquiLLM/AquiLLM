"""Human judgments bind the complete original observation, not repeated text."""

from copy import deepcopy
from pathlib import Path

import pytest

from apps.chat.evals.evidence_quality_eval import (
    digest,
    evaluate,
    load_cases,
    text_digest,
)


def reviewed_case():
    from apps.chat.evals.evidence_effective_config import expected_treatment

    case = load_cases(Path(__file__).parents[1] / "evals/evidence_quality_cases.json")[
        0
    ]
    row = {
        "case_id": case["case_id"],
        "split": case["split"],
        "mode": "combined",
        "backend": "live",
        "invocation_id": "actual-invocation",
        "code_revision": "r",
        "run_id": "run-1",
        "repetition": 1,
        "profile": "quality",
        "answer": "Same short answer.",
        "delivered": [],
        "citations": [],
        "sdk_payloads": [{}],
        "events": [],
        "comparison_controls": {"limit": 1},
        "resolved_treatment": expected_treatment("combined"),
        "snapshot": {
            k: k
            for k in (
                "source",
                "answer",
                "reranker",
                "hardware",
                "embedding",
                "bindings",
            )
        },
    }
    row["snapshot"].update(
        source=digest(case["sources"]), configuration=digest(row["comparison_controls"])
    )
    review = {
        "kind": "human",
        "reviewer": "named-person",
        "answer_sha256": text_digest(row["answer"]),
        "answer_faithful": True,
        "citation_entailment": True,
        "claims": {
            c["claim_id"]: {k: True for k in ["faithful", *c["labels"]]}
            for c in case["required_claims"]
        },
    }
    return case, row, review


def test_answer_hash_alone_is_not_a_review_of_current_evidence():
    case, row, review = reviewed_case()
    assert evaluate(case, row, review)["faithfulness"] is None


@pytest.mark.parametrize(
    "change",
    [
        "mode",
        "run_id",
        "invocation_id",
        "code_revision",
        "repetition",
        "snapshot",
        "delivered",
        "sdk_payloads",
        "events",
    ],
)
def test_repeated_answer_cannot_reuse_review_across_subject_changes(change):
    from apps.chat.evals.evidence_review_subject import review_subject

    case, row, review = reviewed_case()
    review["subject"] = review_subject(row)
    original = evaluate(case, row, review)
    assert original["faithfulness"] == 1 and original["human_review"] == review
    assert (
        evaluate(case, original, original["human_review"])["review_subject"]
        == review["subject"]
    )
    changed = deepcopy(row)
    if change == "snapshot":
        changed[change]["answer"] = "changed-runtime"
    elif change in ("delivered", "sdk_payloads", "events"):
        changed[change] = [] if row[change] else [{"changed": True}]
    elif change == "repetition":
        changed[change] = 2
    else:
        changed[change] = "different"
    # Subject verification occurs before parsing malformed mutated evidence.
    assert review_subject(changed) != review["subject"]
    from apps.chat.evals.evidence_review_subject import validated_review

    assert validated_review(changed, review) is None


def test_comparison_revalidates_human_record_instead_of_cached_scores():
    from apps.chat.evals.evidence_review_subject import (
        quality_review_errors,
        review_subject,
    )

    case, row, review = reviewed_case()
    review["subject"] = review_subject(row)
    result = evaluate(case, row, review)
    reports = {"combined": {"revision": "r", "observations": [result]}}
    assert not quality_review_errors(reports)
    result["sdk_payloads"] = [{"different-evidence": True}]
    assert quality_review_errors(reports)


@pytest.mark.parametrize(
    "field,value",
    [("claims", []), ("claims", {"claim": True}), ("bounded_partial", "yes")],
)
def test_malformed_bound_human_records_remain_unknown(field, value):
    from apps.chat.evals.evidence_review_subject import review_subject

    case, row, review = reviewed_case()
    review["subject"] = review_subject(row)
    review[field] = value
    result = evaluate(case, row, review)
    assert result["faithfulness"] is None
    assert result["human_review"] == review and not result["review_valid"]
