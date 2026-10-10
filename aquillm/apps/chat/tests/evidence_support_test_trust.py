"""Synthetic administrator reader injection; never writes production trust."""

import hashlib
import json

from apps.chat.evals.evidence_quality_eval import digest
from apps.chat.evals.evidence_review_subject import review_subject


def bind_synthetic_delivery(row):
    """Explicit artificial binding for enrollment plumbing, not runtime evidence."""
    row["support_delivery_binding"] = {
        "evidence_digest": "e" * 64,
        "scope_digest": "f" * 64,
        "payload_digest": digest(row["sdk_payloads"][-1]),
        "delivered_digest": digest(row["delivered"]),
    }
    row["support_decision"].update(evidence_digest="e" * 64, scope_digest="f" * 64)
    row["support_packet_identity"] = {
        "evidence_digest": "e" * 64,
        "scope_digest": "f" * 64,
    }
    row["source_scope"]["digest"] = "f" * 64


def install_pins(monkeypatch, calibration, activation=None):
    from apps.chat.evals import evidence_support_trust as trust

    manifest = {
        "schema": "support-review-trust-v1",
        "version": 1,
        "calibration": calibration,
        "activation": activation or {},
    }
    original = trust._read_bytes
    monkeypatch.setattr(
        trust,
        "_read_bytes",
        lambda path, limit: (
            json.dumps(manifest).encode()
            if path == trust.TRUST_PATH
            else original(path, limit)
        ),
    )
    return manifest


def activation_attachment(
    tmp_path, monkeypatch, report, calibration_attachment, policy
):
    from apps.chat.evals.evidence_support_trust import trusted_manifest

    subjects = digest(
        [
            review_subject(r)
            for r in sorted(report["observations"], key=lambda r: r["case_id"])
        ]
    )
    record = {
        "kind": "independent_support_activation",
        "revision": report["revision"],
        "policy_sha256": calibration_attachment["artifact_sha256"],
        "valid_until": policy["valid_until"],
        "subjects_sha256": subjects,
        "observations": report["observations"],
        **{
            k: policy[k]
            for k in (
                "reviewer",
                "provenance_sha256",
                "provenance_artifact",
                "criteria",
                "corpus_sha256",
            )
        },
    }
    path = tmp_path / "activation.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    pin = {
        "revision": report["revision"],
        "policy_sha256": record["policy_sha256"],
        "subjects_sha256": subjects,
        "criteria_sha256": digest(record["criteria"]),
        "reviewer": record["reviewer"],
        "provenance_sha256": record["provenance_sha256"],
    }
    install_pins(monkeypatch, trusted_manifest()["calibration"], {sha: pin})
    return {"artifact": str(path), "artifact_sha256": sha}
