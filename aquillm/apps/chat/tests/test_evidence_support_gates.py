"""Favorable averages never excuse unsupported publication or missing cohorts."""

from copy import deepcopy

import pytest

from apps.chat.evals.evidence_review_subject import review_subject
from apps.chat.evals.evidence_support_gates import (
    paired_domain_errors,
    support_gate_errors,
)
from apps.chat.evals.evidence_support_metrics import measure_cohorts, wilson
from apps.chat.tests.test_evidence_support_policy_records import policy_record


def test_zero_observed_errors_is_not_zero_uncertainty():
    interval = wilson(0, 10, 0.95)
    assert interval["rate"] == 0
    assert 0.27 < interval["upper"] < 0.29


@pytest.mark.parametrize(
    "failure",
    [
        "false_complete",
        "unsupported_answer",
        "unknown_labels",
        "rank_only",
        "publication",
        "delivery",
        "lanes",
    ],
)
def test_current_support_failures_block_even_claimed_policy_verification(
    tmp_path, monkeypatch, failure
):
    record, attachment = policy_record(tmp_path, monkeypatch)
    rows = [r for r in record["observations"] if r["split"] == "heldout"]
    for row in rows:
        row["quality_aggregation"] = "normal"
        row["lane_observations"] = {
            k: {"status": "observed"}
            for k in ("retrieval", "graph", "source_generation")
        }
        row["human_review"]["subject"] = review_subject(row)
    report = {
        "observations": rows,
        "mode": "combined",
        "split": "heldout",
        "corpus_sha256": record["corpus_sha256"],
        "independent_evidence": {"support_policy": attachment},
    }
    assert support_gate_errors(report, {"support_policy_verified": True}) == []
    changed = deepcopy(rows[0])
    if failure in ("false_complete", "unsupported_answer"):
        changed["human_review"]["support_labels"][failure] = True
    elif failure == "unknown_labels":
        changed["human_review"].pop("support_labels")
    elif failure == "rank_only":
        changed["support_decision"] = {
            "state": "complete",
            "assessment_basis": "unvalidated",
            "normalized_relevance": 1.0,
        }
    elif failure == "publication":
        changed["final_stream"] = "unsupported post-repair text"
    elif failure == "delivery":
        changed["support_delivery_binding"] = {}
    elif failure == "lanes":
        changed["lane_observations"] = {}
    rows[0] = changed
    errors = support_gate_errors(report, {"support_policy_verified": True})
    expected = {
        "false_complete": "independently observed unsupported completion/answer",
        "unsupported_answer": "independently observed unsupported completion/answer",
        "unknown_labels": "missing independent support labels",
        "rank_only": "unsupported complete decision",
        "publication": "support final publication missing or mismatched",
        "delivery": "support final delivery binding missing or mismatched",
        "lanes": "support retrieval/graph/source-generation observations missing",
    }
    assert expected[failure] in errors


def test_selectively_favorable_cohort_cannot_replace_missing_samples(
    tmp_path, monkeypatch
):
    record, _ = policy_record(tmp_path, monkeypatch)
    rows = [r for r in record["observations"] if r["depth"] == "routine"]
    with pytest.raises(ValueError, match="missing independently labeled"):
        measure_cohorts(rows, record["criteria"])


def test_pooled_quality_gain_cannot_hide_domain_failure():
    from apps.chat.evals.evidence_quality_eval import MODES

    groups = [
        {
            mode: {
                "domain": domain,
                "support_recall": score if mode != "baseline" else 0.5,
                "faithfulness": 1,
                "qualification_accuracy": 1,
            }
            for mode in MODES
        }
        for domain, score in (("manual", 0.25), ("research", 1.0))
    ]
    assert any("manual/support_recall" in e for e in paired_domain_errors(groups))


def crossing_cohort_reports(metric="support_recall", arm="combined", decline=True):
    """Use the frozen held-out domain/depth membership without changing labels."""
    from pathlib import Path

    from apps.chat.evals import evidence_quality_eval as evaluation
    from apps.chat.tests.test_evidence_quality_gates import reports

    cases = evaluation.load_cases(
        Path(evaluation.__file__).with_name("evidence_quality_cases.json")
    )
    result = reports()
    for mode, report in result.items():
        template = report["observations"][0]
        report["observations"] = []
        for case in cases:
            if case["split"] != "heldout" or case["quality_aggregation"] != "normal":
                continue
            row = {
                **deepcopy(template),
                "case_id": case["case_id"],
                "depth": case["depth"],
                "domain": case["genre"],
                "scenario": case["scenario"],
                **{
                    key: 0.0 if mode == "baseline" else 1.0
                    for key in (
                        "support_recall",
                        "faithfulness",
                        "qualification_accuracy",
                    )
                },
            }
            if case["genre"] == "policy_record" and case["depth"] == "deeper":
                row[metric] = 1.0
                if decline and mode == arm and case["case_id"] == "ep-hel-pol-03":
                    row[metric] = 0.8
            report["observations"].append(row)
    return result


@pytest.mark.parametrize(
    "metric", ["support_recall", "faithfulness", "qualification_accuracy"]
)
@pytest.mark.parametrize("arm", ["selection", "preservation", "combined"])
def test_crossing_cohort_decline_blocks_passing_domain_and_depth_marginals(metric, arm):
    from apps.chat.evals.evidence_quality_comparison import paired_bootstrap_interval
    from apps.chat.evals.evidence_quality_eval import join_arms

    reports = crossing_cohort_reports(metric, arm)
    groups = join_arms({m: r["observations"] for m, r in reports.items()})
    policy = [g for g in groups if g["baseline"]["domain"] == "policy_record"]
    deeper = [g for g in groups if g["baseline"]["depth"] == "deeper"]
    crossed = [g for g in policy if g["baseline"]["depth"] == "deeper"]
    assert len(policy) == 8 and len(crossed) == 3
    for cohort in (groups, policy, deeper):
        assert (
            paired_bootstrap_interval(
                [g[arm][metric] - g["baseline"][metric] for g in cohort]
            )["lower"]
            >= 0
        )
    assert (
        paired_bootstrap_interval(
            [g[arm][metric] - g["baseline"][metric] for g in crossed]
        )["lower"]
        < 0
    )
    assert paired_domain_errors(groups) == [
        f"{arm}: domain policy_record/deeper/{metric} decline or unknown"
    ]


@pytest.mark.parametrize(
    "missing", [None, "baseline_metric", "candidate_metric", "empty"]
)
def test_intersections_accept_nonregression_and_reject_unknown_or_empty(missing):
    from apps.chat.evals.evidence_quality_eval import join_arms

    reports = crossing_cohort_reports(decline=False)
    groups = join_arms({m: r["observations"] for m, r in reports.items()})
    assert paired_domain_errors(groups) == []
    if missing == "empty":
        groups = [
            g
            for g in groups
            if not (
                g["baseline"]["domain"] == "policy_record"
                and g["baseline"]["depth"] == "deeper"
            )
        ]
    elif missing:
        group = next(g for g in groups if g["baseline"]["case_id"] == "ep-hel-pol-03")
        mode = "baseline" if missing == "baseline_metric" else "combined"
        group[mode]["support_recall"] = None
    errors = paired_domain_errors(groups)
    if missing:
        assert (
            "combined: domain policy_record/deeper/support_recall decline or unknown"
            in errors
        )
    else:
        assert errors == []
