"""Independent support decisions must bind actual delivery and fail closed."""

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from apps.chat.evals.evidence_review_subject import review_subject, validated_review
from apps.chat.tests.test_evidence_review_subject import reviewed_case


@pytest.mark.parametrize(
    "field",
    [
        "primary_question",
        "source_scope",
        "support_decision",
        "support_delivery_binding",
        "support_packet_identity",
        "runtime_identity",
        "policy_identity",
        "final_stream",
        "persisted_answer",
        "corpus_sha256",
    ],
)
def test_changed_support_subject_invalidates_review(field):
    _, row, review = reviewed_case()
    row.update(
        primary_question="What is the amount?",
        source_scope={"kind": "selected_documents"},
        support_decision={"state": "unknown", "assessment_basis": "unvalidated"},
        runtime_identity="fixture-model-v1",
        policy_identity=None,
        final_stream=row["answer"],
        persisted_answer=row["answer"],
        corpus_sha256="frozen",
    )
    review["subject"] = review_subject(row)
    assert validated_review(row, review)
    changed = deepcopy(row)
    changed[field] = "changed"
    assert validated_review(changed, review) is None


def test_missing_support_evidence_blocks_otherwise_favorable_comparison(monkeypatch):
    from apps.chat.evals import evidence_quality_gates as gates
    from apps.chat.tests.test_evidence_quality_gates import TARGETS, reports

    monkeypatch.setattr(gates, "frozen_comparison_errors", lambda _: [])
    monkeypatch.setattr(gates, "verified_flags", lambda r: r)
    monkeypatch.setattr(
        "apps.chat.evals.evidence_review_subject.quality_review_errors", lambda _: []
    )
    monkeypatch.setattr(
        "apps.chat.evals.evidence_operational.validate_attachment",
        lambda *a: {"passed": True, "blocking_reasons": []},
    )
    result = gates.compare(reports(), TARGETS)
    assert not result["activation_eligible"]
    assert any("support" in reason for reason in result["blocking_reasons"])


def test_self_assessed_support_does_not_activate_policy():
    from apps.chat.evals.evidence_support_calibration import load_support_policy

    assert (
        load_support_policy(
            {"kind": "model_self_assessment", "all_answers_supported": True},
            runtime_identity="fixture-model-v1",
            scope_kind="selected_documents",
        )
        is None
    )


def test_exact_baseline_fixture_command_persists_before_exit_two():
    command = [
        sys.executable,
        "aquillm/apps/chat/evals/run_evidence_quality_eval.py",
        "--backend",
        "fixture",
        "--mode",
        "baseline",
        "--split",
        "development",
        "--report",
        "artifacts/rag-remediation/a3-fixture.json",
        "--require-activation",
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode == 2, result.stderr
    report = json.loads(Path(command[-2]).read_text(encoding="utf-8"))
    assert report["backend"] == "fixture"
    assert report["activation_eligible"] is False
    assert len(report["observations"]) == 40
    print(
        "A3 CLI: "
        + " ".join(command)
        + "; exit=2; persisted observations=40; activation=false"
    )


def test_supplement_labels_are_frozen_and_cannot_mint_human_evidence(tmp_path):
    from apps.chat.evals.evidence_support_cases import load_support_cases

    cases = load_support_cases()
    assert len(cases) == 32
    path = tmp_path / "mutated.json"
    path.write_text(
        json.dumps(
            {
                "cases": cases,
                "independent_human_labels": False,
                "activation_eligible": False,
            }
        )
    )
    root = json.loads(path.read_text())
    root["cases"][-1]["expected"] = "complete"
    path.write_text(json.dumps(root))
    with pytest.raises(ValueError, match="frozen"):
        load_support_cases(path)


@pytest.mark.parametrize(
    "scenario",
    [
        "unrelated_citation",
        "mixed_bullets",
        "mixed_prose",
        "missing_comparison",
        "negative_claim",
        "exhaustive_claim",
        "contradiction",
        "numerical_mismatch",
        "unit_mismatch",
        "tail_loss",
        "boundary_loss",
        "support_removed",
        "retrieved_instructions",
        "revoked_source",
        "no_result_outage",
        "memory_as_document",
    ],
)
def test_adversarial_supplement_self_assessment_never_approves(scenario):
    from apps.chat.evals.evidence_support_calibration import load_support_policy
    from apps.chat.evals.evidence_support_cases import load_support_cases

    for case in load_support_cases():
        if case["scenario"] == scenario:
            assert (
                load_support_policy(
                    {
                        **case,
                        "kind": "model_self_assessment",
                        "supported": True,
                        "normalized_relevance": 1.0,
                        "score_status": "uncalibrated",
                    },
                    runtime_identity="fixture-model-v1",
                    scope_kind="selected_documents",
                )
                is None
            )
