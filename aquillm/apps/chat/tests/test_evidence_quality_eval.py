"""Frozen source spans, explicit human review, and paired rollout gates."""

import copy
from pathlib import Path

import pytest

from apps.chat.evals import evidence_quality_eval as quality

CASES = Path(__file__).parents[1] / "evals/evidence_quality_cases.json"


def case(index=0):
    return quality.load_cases(CASES)[index]


def delivered(c):
    return [
        dict(
            source_id=s["source_id"],
            revision=s["revision"],
            fingerprint=quality.text_digest(s["text"]),
            start=0,
            end=len(s["text"]),
            text=s["text"],
        )
        for s in c["sources"]
        if s["authorized_at_answer"]
    ]


def test_frozen_corpus_and_exact_span_omission():
    cases = quality.load_cases(CASES)
    assert len(cases) == 80
    c = case(2)
    assert (
        quality.support_recall(
            quality.required_support(c), quality.match_delivered(c, delivered(c)[:-1])
        )
        == 0.8
    )
    span = delivered(case())[0]
    span.update(text=span["text"][:1000], end=1000)
    assert not quality.match_delivered(case(), [span])


@pytest.mark.parametrize(
    "field,value",
    [("revision", "stale"), ("fingerprint", "wrong"), ("source_id", "other")],
)
def test_same_quote_wrong_identity_is_not_delivery(field, value):
    c = case()
    spans = delivered(c)
    spans[0][field] = value
    assert not quality.match_delivered(c, spans)


def test_optional_context_and_ordinal_packet_oracle_are_not_required():
    for index in (1, 4):
        c = case(index)
        required = quality.required_support(c)
        spans = []
        for gold in c["gold_support"]:
            if gold["support_id"] in required:
                s = next(s for s in c["sources"] if s["source_id"] == gold["source_id"])
                spans.append(
                    {
                        **gold,
                        "text": gold["quote"],
                        "fingerprint": quality.text_digest(s["text"]),
                    }
                )
        assert quality.support_recall(required, quality.match_delivered(c, spans)) == 1


@pytest.mark.parametrize(
    "index,field",
    [
        (0, "unit"),
        (0, "quantity"),
        (0, "condition"),
        (1, "negation"),
        (1, "controlling_date"),
    ],
)
def test_human_review_catches_qualifications_and_unknown(index, field):
    c = case(index)
    assert quality.review_claims(c, None)["faithfulness"] is None
    claims = {
        x["claim_id"]: {k: True for k in ["faithful", *x["labels"]]}
        for x in c["required_claims"]
    }
    claims[c["required_claims"][0]["claim_id"]][field] = False
    review = {"reviewer": "human-1", "kind": "human", "claims": claims}
    assert quality.review_claims(c, review)["qualification_accuracy"] < 1
    review["kind"] = "model"
    assert quality.review_claims(c, review)["faithfulness"] is None


def test_citations_require_current_delivered_source_including_optional():
    c = case()
    citations = c["permitted_citations"]
    assert quality.citation_violations(c, citations, delivered(c)) == []
    assert quality.citation_violations(c, citations, []) == citations
    assert quality.citation_violations(
        c, [{"source_id": "fabricated", "revision": "r"}], delivered(c)
    )


def rows(mode):
    return [
        {
            "case_id": str(i),
            "split": "development",
            "mode": mode,
            "snapshot": {
                "source": str(i),
                "answer": "a",
                "reranker": "r",
                "hardware": "h",
            },
            "support_recall": i / 2,
        }
        for i in range(2)
    ]


def test_four_arm_join_uses_ids_rejects_missing_duplicates_and_drift():
    arms = {mode: rows(mode) for mode in quality.MODES}
    arms["combined"].reverse()
    assert len(quality.join_arms(arms)) == 2
    for mutation in ("missing", "duplicate", "drift"):
        bad = copy.deepcopy(arms)
        if mutation == "missing":
            bad["combined"].pop()
        elif mutation == "duplicate":
            bad["combined"].append(bad["combined"][0])
        else:
            bad["combined"][0]["snapshot"]["answer"] = "changed"
        with pytest.raises(ValueError):
            quality.join_arms(bad)


def test_no_denominator_no_grades_and_safety_are_unknown_not_pass():
    assert quality.support_recall(set(), set()) is None
    assert quality.ndcg([], None) is None
    assert quality.score_safety(case(8), {}) is None
    assert (
        quality.score_safety(
            case(8),
            {
                "closed": True,
                "late_publications": 1,
                "answer": "",
                "delivered": [],
                "citations": [],
            },
        )
        is False
    )


def test_fixture_report_cannot_activate(tmp_path):
    import json

    from apps.chat.evals.run_evidence_quality_eval import main

    report = tmp_path / "report.json"
    assert (
        main(
            [
                "--cases",
                str(CASES),
                "--report",
                str(report),
                "--split",
                "development",
                "--mode",
                "combined",
                "--backend",
                "fixture",
            ]
        )
        == 0
    )
    result = json.loads(report.read_text())
    assert result["activation_eligible"] is False
    assert len(result["observations"]) == 40
    assert all(x["faithfulness"] is None for x in result["observations"])
