"""Activation rejects unknowns, changed models and poor quality/responsiveness."""

from copy import deepcopy

import pytest

from apps.chat.evals.evidence_quality_eval import MODES
from apps.chat.evals.evidence_quality_gates import compare


@pytest.fixture(autouse=True)
def independent_operational_gate(monkeypatch):
    # Metric unit tests isolate the separately tested attachment validator.
    monkeypatch.setattr(
        "apps.chat.evals.evidence_operational.validate_attachment",
        lambda *a: {"passed": True, "blocking_reasons": []},
    )
    monkeypatch.setattr(
        "apps.chat.evals.evidence_quality_gates.frozen_comparison_errors", lambda *a: []
    )
    monkeypatch.setattr(
        "apps.chat.evals.evidence_quality_gates.verified_flags", lambda r: r
    )


def reports():
    result = {}
    for mode in MODES:
        rows = []
        for i, depth in enumerate(("routine", "deeper")):
            rows.append(
                {
                    "case_id": str(i),
                    "split": "heldout",
                    "mode": mode,
                    "snapshot": {
                        "source": str(i),
                        "answer": "a",
                        "reranker": "r",
                        "hardware": "h",
                    },
                    "quality_aggregation": "normal",
                    "scenario": "tail",
                    "depth": depth,
                    "support_recall": 0.5 if mode in ("baseline", "selection") else 1,
                    "faithfulness": 1,
                    "qualification_accuracy": 1,
                    "ndcg": None,
                    "citation_violations": [],
                    "provenance_complete": True,
                    "dispatch_accounting_complete": True,
                    "actual_safety": {"passed": True},
                    "timings_ms": {"first_grounded": 20, "completion": 30},
                }
            )
        rows.append(
            {
                **rows[0],
                "case_id": "safety",
                "quality_aggregation": "safety_only",
                "safety": mode in ("preservation", "combined"),
            }
        )
        result[mode] = {
            "backend": "live",
            "dirty": False,
            "pair_capability_verified": True,
            "observations": rows,
            **{
                k: True
                for k in (
                    "runtime_verified",
                    "corpus_human_reviewed",
                    "deterministic_regressions_passed",
                    "concurrent_load_verified",
                    "cold_warm_verified",
                    "completion_reserve_measured",
                )
            },
        }
    return result


TARGETS = {
    depth: {"first_grounded": 25, "completion": 35} for depth in ("routine", "deeper")
}


def test_incomplete_frozen_corpus_and_unmatched_conditions_rejected():
    from apps.chat.evals.evidence_quality_comparison import frozen_comparison_errors

    assert frozen_comparison_errors(reports())


def test_no_operational_evidence_still_blocks_the_full_gate(monkeypatch):
    from apps.chat.evals import evidence_operational

    monkeypatch.undo()
    result = compare(reports(), TARGETS)
    assert not result["activation_eligible"]
    assert any("pair_or_deadline unproven" in r for r in result["blocking_reasons"])
    assert not evidence_operational.validate_attachment([], reports())["passed"]


def test_tail_latency_interval_exposes_uncertainty():
    from apps.chat.evals.evidence_quality_comparison import p95_ratio_interval

    interval = p95_ratio_interval([10, 20, 30], [30, 10, 20])
    assert interval["ratio"] == 1 and interval["upper"] > 1


def test_frozen_join_rejects_matching_but_wrong_source_snapshot():
    from pathlib import Path

    from apps.chat.evals.evidence_quality_comparison import frozen_comparison_errors
    from apps.chat.evals.evidence_quality_eval import FROZEN_SHA256, digest, load_cases

    cases = [
        c
        for c in load_cases(
            Path(__file__).parents[1] / "evals/evidence_quality_cases.json"
        )
        if c["split"] == "development"
    ]
    reports = {
        mode: {
            "split": "development",
            "corpus_sha256": FROZEN_SHA256,
            "concurrency": 2,
            "cache_state": "cold",
            "observations": [
                {**c, "snapshot": {"source": digest(c["sources"])}} for c in cases
            ],
        }
        for mode in MODES
    }
    assert frozen_comparison_errors(reports) == []
    for report in reports.values():
        report["observations"][0]["snapshot"]["source"] = "identical-but-wrong"
    assert frozen_comparison_errors(reports)


def test_baseline_new_safety_contract_is_diagnostic_but_candidate_unknown_blocks():
    report = reports()
    assert compare(report, TARGETS)["activation_eligible"]
    for mode in ("preservation", "combined"):
        for value in (None, False):
            changed = deepcopy(report)
            changed[mode]["observations"][-1]["actual_safety"]["passed"] = value
            assert not compare(changed, TARGETS)["activation_eligible"]


def test_baseline_ordinary_citation_violation_still_blocks():
    report = reports()
    report["baseline"]["observations"][0]["citation_violations"] = ["wrong"]
    assert not compare(report, TARGETS)["activation_eligible"]


def test_fixture_unknown_human_judgment_and_latency_regression_block():
    for mutation in ("fixture", "human", "latency", "targets", "no_gain"):
        report = reports()
        if mutation == "fixture":
            report["combined"]["backend"] = "fixture"
        elif mutation == "human":
            report["combined"]["observations"][0]["faithfulness"] = None
        elif mutation == "latency":
            report["combined"]["observations"][0]["timings_ms"]["completion"] = 31
        elif mutation == "no_gain":
            report["combined"]["observations"][0]["support_recall"] = 0.5
            report["combined"]["observations"][1]["support_recall"] = 0.5
        assert not compare(report, None if mutation == "targets" else TARGETS)[
            "activation_eligible"
        ]
