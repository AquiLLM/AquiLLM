"""Actual effective capture and current comparator attachment boundaries."""

from copy import deepcopy

import pytest

from apps.knowledge_graph.evals.graph_activation_attachment_v2 import (
    graph_activation_attachment_v2,
)
from apps.knowledge_graph.evals.graph_activation_attachment_v2 import (
    validate_graph_activation_attachment_v2 as validate,
)
from apps.knowledge_graph.evals.graph_effective_controls import graph_effective_controls
from apps.knowledge_graph.tests.graph_gate_test_fixtures import digest, fixture_report
from apps.knowledge_graph.tests.graph_review_fixtures import rebind


def reports(control=None):
    controls = {"graph_policy": control or graph_effective_controls()}
    return {
        mode: {
            "revision": "a" * 40,
            "observations": [
                {
                    "comparison_controls": deepcopy(controls),
                    "snapshot": {"configuration": digest(controls)},
                    "lane_observations": {
                        "graph": {
                            "status": "observed",
                            "runtime_sha256": controls["graph_policy"][
                                "runtime_sha256"
                            ],
                            "source_generation_sha256": digest("source_generation"),
                        }
                    },
                }
            ],
        }
        for mode in ("baseline", "selection", "preservation", "combined")
    }


def test_actual_capture_unchanged_legacy_preserves_source_window_evaluation():
    from apps.knowledge_graph.evals.run_kg_eval import (
        current_comparison_algorithm_signatures,
    )

    captured = graph_effective_controls()
    assert (
        captured["algorithm_signature"]
        == current_comparison_algorithm_signatures()["ppr_v1"]
    )
    result = validate(None, quality_reports=reports(captured))
    assert result["passed"], result
    assert result["status"] == "not_applicable_unchanged_legacy"


@pytest.mark.parametrize(
    "mutation", ["literal_legacy", "config", "unknown", "runtime", "lane_missing"]
)
def test_legacy_strings_or_changed_effective_identity_cannot_opt_out(mutation):
    quality = reports()
    row = quality["combined"]["observations"][0]
    policy = row["comparison_controls"]["graph_policy"]
    if mutation == "literal_legacy":
        policy["algorithm_signature"] = "legacy"
    elif mutation == "config":
        policy["algorithm_config"]["max_candidates"] = 19
    elif mutation == "unknown":
        del row["comparison_controls"]["graph_policy"]
    elif mutation == "runtime":
        policy["runtime_sha256"] = "b" * 64
    else:
        del row["lane_observations"]["graph"]
    assert not validate(None, quality_reports=quality)["passed"]


def v2_reports(monkeypatch):
    from django.conf import settings

    monkeypatch.setattr(settings, "KG_GRAPH_REMEDIATION_MODE", "v2", raising=False)
    control = graph_effective_controls()
    assert control["remediation_mode"] == "v2"
    return reports(control)


def test_equal_v2_across_four_arms_still_requires_graph_report(monkeypatch):
    result = validate(None, quality_reports=v2_reports(monkeypatch))
    assert not result["passed"]
    assert "graph_report_missing" in result["blocking_reasons"]


def test_equal_adaptive_ranking_cannot_use_unchanged_legacy_exception(monkeypatch):
    from django.conf import settings

    monkeypatch.setattr(settings, "KG_PPR_RESTART_MODE", "adaptive")
    result = validate(None, quality_reports=reports())
    assert not result["passed"]
    assert "graph_report_missing" in result["blocking_reasons"]


def attachment_pair(monkeypatch, control_override=None):
    quality = v2_reports(monkeypatch)
    if control_override is not None:
        quality = reports(control_override)
    report = fixture_report()
    control = quality["combined"]["observations"][0]["comparison_controls"][
        "graph_policy"
    ]
    model_runtime = {
        k: digest(k) for k in ("answer", "reranker", "hardware", "embedding")
    }
    for binding in (report["binding"], report["baseline_binding"]):
        binding["model_runtime"] = digest(model_runtime)
        binding["budget"] = digest({})
    report["controls"]["candidate"] = deepcopy(control)
    rebind(report)
    source_scope = {"kind": "synthetic", "sources": ["source"]}
    for pair in report["queries"][0]["executions"].values():
        for arm in ("baseline", "candidate"):
            pair[arm]["inputs"]["scope_sha256"] = digest(source_scope)
    attached = graph_activation_attachment_v2(report)
    for mode, arm in quality.items():
        row = arm["observations"][0]
        row["snapshot"].update(model_runtime)
        row["source_scope"] = source_scope
        query = report["queries"][0]
        candidate = query["executions"][mode]["candidate"]
        row.update(
            code_revision="a" * 40,
            primary_question=candidate["inputs"]["question"],
        )
        row["lane_observations"]["graph"].update(
            query_id=query["query_id"],
            observation_sha256=digest(query),
            execution_sha256=digest(candidate),
            mode=mode,
            request_algorithm_signature=candidate["request_algorithm_signature"],
            attachment_sha256=attached["attachment_sha256"],
            corpus_sha256=report["binding"]["corpus"],
            budget_sha256=report["binding"]["budget"],
            policy_sha256=attached["policy_sha256"],
            frontier_budget=deepcopy(candidate["frontier_budget"]),
            frontier_invocations=deepcopy(candidate["frontier_invocations"]),
            requested_subset_signature=candidate["frontier_invocations"][0][
                "requested_subset_signature"
            ],
        )
    return report, quality


def test_required_attachment_binds_actual_subject_and_mutations(monkeypatch):
    report, quality = attachment_pair(monkeypatch)
    assert validate(report, quality_reports=quality)["passed"]
    for key in (
        "attachment_sha256",
        "observation_sha256",
        "source_generation_sha256",
        "runtime_sha256",
        "policy_sha256",
        "budget_sha256",
        "corpus_sha256",
        "frontier_budget",
        "frontier_invocations",
        "requested_subset_signature",
        "execution_sha256",
        "mode",
        "request_algorithm_signature",
    ):
        changed = deepcopy(quality)
        changed["combined"]["observations"][0]["lane_observations"]["graph"][key] = (
            "f" * 64
        )
        assert not validate(report, quality_reports=changed)["passed"], key


def test_real_comparator_adds_graph_block_without_bypassing_other_reasons():
    from apps.chat.evals.evidence_quality_gates import compare
    from apps.chat.tests.test_evidence_quality_gates import TARGETS
    from apps.chat.tests.test_evidence_quality_gates import reports as quality_reports

    result = compare(quality_reports(), TARGETS, graph_report=fixture_report())
    assert result["activation_eligible"] is False
    assert "graph_applicability_unknown" in result["blocking_reasons"]
    assert any("pair_or_deadline" in r for r in result["blocking_reasons"])
