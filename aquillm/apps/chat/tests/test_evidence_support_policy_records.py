"""Synthetic enrollment tests are plumbing evidence, never human validation."""

import hashlib
import json
from copy import deepcopy
from datetime import UTC, datetime, timedelta

import pytest

from apps.chat.evals.evidence_quality_eval import digest
from apps.chat.evals.evidence_review_subject import review_subject
from apps.chat.tests.evidence_support_test_trust import (
    activation_attachment,
    bind_synthetic_delivery,
    install_pins,
)
from apps.chat.tests.test_evidence_review_subject import reviewed_case


def policy_record(tmp_path, monkeypatch):

    rows = []
    for split in ("development", "heldout"):
        for depth in ("routine", "deeper"):
            for i in range(10):
                _, row, review = reviewed_case()
                row.update(
                    case_id=f"{split}-{depth}-{i}",
                    split=split,
                    depth=depth,
                    domain="synthetic",
                    runtime_identity="runtime-v1",
                    primary_question="What is the amount?",
                    source_scope={"kind": "selected_documents"},
                    support_decision={
                        "state": "unknown",
                        "assessment_basis": "unvalidated",
                    },
                    policy_identity="policy-v1",
                    final_stream=row["answer"],
                    persisted_answer=row["answer"],
                    final_delta=row["answer"],
                    final_message_id=str(i),
                    corpus_sha256="a" * 64,
                    support_question="What is the amount?",
                    publication_capture_complete=True,
                    provenance_complete=True,
                    dispatch_accounting_complete=True,
                    source_validation_complete=True,
                    publication_observation_complete=True,
                )
                bind_synthetic_delivery(row)
                review["subject"] = review_subject(row)
                review["support_labels"] = {
                    "quote_accuracy": True,
                    "source_entailment": True,
                    "inference": True,
                    "numerical_conditions": True,
                    "contradiction_free": True,
                    "coverage": True,
                    "false_complete": False,
                    "unsupported_answer": False,
                    "abstained": i % 2 == 0,
                    "abstention_correct": True,
                }
                row["human_review"] = review
                rows.append(row)
    now = datetime.now(UTC)
    provenance = tmp_path / "provenance.txt"
    provenance.write_text("Synthetic enrollment exercise; not a real human review.")
    record = {
        "kind": "independent_support_policy",
        "policy_id": "policy-v1",
        "runtime_identity": "runtime-v1",
        "scope_kind": "selected_documents",
        "decision_version": "support-v1",
        "model_identity": "model-v1",
        "tokenizer_identity": "tokenizer-v1",
        "input_preparation_identity": "input-v1",
        "corpus_sha256": "a" * 64,
        "reviewer": "synthetic-person",
        "provenance_sha256": hashlib.sha256(provenance.read_bytes()).hexdigest(),
        "provenance_artifact": str(provenance),
        "valid_until": (now + timedelta(days=1)).isoformat(),
        "predeclared_at": (now - timedelta(days=2)).isoformat(),
        "heldout_opened_at": (now - timedelta(days=1)).isoformat(),
        "criteria": {
            "method": "wilson",
            "modes": ["combined"],
            "confidence": 0.95,
            "minimum_samples": 10,
            "domains": ["synthetic"],
            "depths": ["routine", "deeper"],
            "max_false_complete": 0.3,
            "max_unsupported_answer": 0.3,
            "min_abstention_quality": 0.5,
            "min_answer_coverage": 0.2,
        },
        "observations": rows,
        "split_sha256": {
            split: digest([review_subject(r) for r in rows if r["split"] == split])
            for split in ("development", "heldout")
        },
    }
    artifact = tmp_path / "review.json"
    artifact.write_text(json.dumps(record), encoding="utf-8")
    sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
    install_pins(
        monkeypatch,
        {
            sha: {
                "reviewer": record["reviewer"],
                "provenance_sha256": record["provenance_sha256"],
                "criteria_sha256": digest(record["criteria"]),
                "subjects_sha256": digest(record["split_sha256"]),
            }
        },
    )
    return record, {"artifact": str(artifact), "artifact_sha256": sha}


def test_enrolled_policy_plumbing_and_runtime_scope_expiry_checks(
    tmp_path, monkeypatch
):
    from apps.chat.evals.evidence_support_calibration import load_support_policy

    _, attachment = policy_record(tmp_path, monkeypatch)
    policy = load_support_policy(
        attachment, runtime_identity="runtime-v1", scope_kind="selected_documents"
    )
    assert policy is not None and policy.policy_id == "policy-v1"
    assert (
        load_support_policy(
            attachment, runtime_identity="changed", scope_kind="selected_documents"
        )
        is None
    )
    assert (
        load_support_policy(
            attachment, runtime_identity="runtime-v1", scope_kind="all_documents"
        )
        is None
    )
    install_pins(monkeypatch, {})
    assert (
        load_support_policy(
            attachment, runtime_identity="runtime-v1", scope_kind="selected_documents"
        )
        is None
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "cohort",
        "sample",
        "uncertainty",
        "expired",
        "predeclared",
        "review",
        "corpus",
        "sink",
        "false_complete",
        "nan",
        "boolean",
    ],
)
def test_enrollment_digest_alone_does_not_approve_bad_measurements(
    tmp_path, monkeypatch, mutation
):
    from apps.chat.evals import evidence_support_calibration as calibration

    record, attachment = policy_record(tmp_path, monkeypatch)
    if mutation == "cohort":
        record["criteria"]["domains"].append("missing")
    elif mutation == "sample":
        record["criteria"]["minimum_samples"] = 11
    elif mutation == "uncertainty":
        record["criteria"]["max_false_complete"] = 0.01
    elif mutation == "expired":
        record["valid_until"] = "2000-01-01T00:00:00+00:00"
    elif mutation == "predeclared":
        record["predeclared_at"] = record["heldout_opened_at"]
    elif mutation == "review":
        record["observations"][0]["human_review"]["kind"] = "model"
    elif mutation == "corpus":
        record["observations"][0]["corpus_sha256"] = "c" * 64
    elif mutation == "sink":
        record["observations"][0]["persisted_answer"] = "different"
    elif mutation == "false_complete":
        record["observations"][0]["human_review"]["support_labels"][
            "false_complete"
        ] = True
    else:
        record["criteria"]["confidence"] = float("nan") if mutation == "nan" else True
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    from apps.chat.evals.evidence_support_trust import trusted_manifest

    pin = deepcopy(next(iter(trusted_manifest()["calibration"].values())))
    pin["criteria_sha256"] = (
        digest(record["criteria"]) if mutation != "nan" else "c" * 64
    )
    install_pins(monkeypatch, {sha: pin})
    assert (
        calibration.load_support_policy(
            {"artifact": str(path), "artifact_sha256": sha},
            runtime_identity="runtime-v1",
            scope_kind="selected_documents",
        )
        is None
    )


def test_existing_attachment_requires_exact_enrolled_subject_and_human_labels(
    tmp_path, monkeypatch
):
    from apps.chat.evals.evidence_quality_review import attach_evidence

    record, attachment = policy_record(tmp_path, monkeypatch)
    rows = deepcopy(record["observations"])
    for row in rows:
        row["code_revision"] = "d" * 40
        row["human_review"]["subject"] = review_subject(row)
    report = {
        "backend": "live",
        "revision": "d" * 40,
        "corpus_sha256": record["corpus_sha256"],
        "observations": rows,
    }
    evidence = {
        "kind": "human_operator",
        "reviewer": "synthetic-person",
        "revision": "d" * 40,
        "corpus_sha256": record["corpus_sha256"],
        "snapshot_digest": digest(
            [r["snapshot"] for r in sorted(rows, key=lambda r: r["case_id"])]
        ),
        "support_subjects_sha256": digest(
            [review_subject(r) for r in sorted(rows, key=lambda r: r["case_id"])]
        ),
        "support_policy": attachment,
        "support_activation": activation_attachment(
            tmp_path, monkeypatch, report, attachment, record
        ),
    }
    assert attach_evidence(report, evidence)["support_policy_verified"] is True
    for field in (
        "primary_question",
        "final_stream",
        "persisted_answer",
        "delivered",
        "source_scope",
        "runtime_identity",
        "policy_identity",
    ):
        changed = deepcopy(report)
        changed["observations"][0][field] = "changed"
        assert attach_evidence(changed, evidence)["support_policy_verified"] is False
    changed = deepcopy(report)
    changed["observations"][0]["human_review"]["support_labels"]["coverage"] = False
    assert attach_evidence(changed, evidence)["support_policy_verified"] is False
    report["backend"] = "fixture"
    assert attach_evidence(report, evidence)["support_policy_verified"] is False


def test_changed_private_artifact_revokes_enrollment(tmp_path, monkeypatch):
    from apps.chat.evals.evidence_support_calibration import load_support_policy

    record, attachment = policy_record(tmp_path, monkeypatch)
    from pathlib import Path

    Path(record["provenance_artifact"]).write_text("changed provenance")
    assert (
        load_support_policy(
            attachment, runtime_identity="runtime-v1", scope_kind="selected_documents"
        )
        is None
    )
