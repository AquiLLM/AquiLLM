"""Attach operator/human evidence to immutable observations without rerunning models."""

import hashlib
from pathlib import Path

from .evidence_quality_eval import digest


def verified_flags(report):
    try:
        return attach_evidence(report, report.get("independent_evidence"))
    except (OSError, ValueError, KeyError, TypeError):
        return {}


def dirty_checkout():
    import subprocess

    return bool(
        subprocess.run(
            ["rtk", "git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )


def attach_evidence(report, evidence):
    """Independent records are required, never a model's self-assessment.

    Artifact hashes let reviewers inspect the measurements behind each claim.
    This validates binding/shape; the named human/operator owns their accuracy.
    """
    flags = (
        "runtime_verified",
        "corpus_human_reviewed",
        "deterministic_regressions_passed",
        "completion_reserve_measured",
        "cold_warm_verified",
    )
    result = dict.fromkeys(flags, False)
    if (
        not evidence
        or evidence.get("kind") != "human_operator"
        or not evidence.get("reviewer")
    ):
        return result
    if (
        evidence.get("revision") != report["revision"]
        or evidence.get("corpus_sha256") != report["corpus_sha256"]
    ):
        raise ValueError("rollout evidence revision/corpus mismatch")
    if evidence.get("snapshot_digest") != digest(
        [
            r["snapshot"]
            for r in sorted(report["observations"], key=lambda r: r["case_id"])
        ]
    ):
        raise ValueError("rollout evidence snapshot mismatch")
    for flag in flags:
        record = evidence.get(flag)
        if not record:
            continue
        if not record.get("reviewed_by") or record.get("status") != "passed":
            continue
        artifact = Path(record["artifact"])
        if (
            hashlib.sha256(artifact.read_bytes()).hexdigest()
            != record["artifact_sha256"]
        ):
            raise ValueError("rollout measurement artifact changed")
        if flag == "completion_reserve_measured":
            configured = record.get("configured_reserve_ms", 0)
            if (
                not 0 < record.get("authorization_packet_p95_ms", 0) <= configured
                or not report["observations"]
                or any(
                    row.get("comparison_controls", {}).get("completion_reserve_ms")
                    != configured
                    for row in report["observations"]
                )
            ):
                continue
        result[flag] = True
    return {**result, "independent_evidence": evidence}


def load_observations(report, cases, *, backend, mode):
    if report["backend"] != backend or report["mode"] != mode:
        raise ValueError("observation mode/backend mismatch")
    indexed = {r["case_id"]: r for r in report["observations"]}
    if len(indexed) != len(report["observations"]) or set(indexed) != {
        c["case_id"] for c in cases
    }:
        raise ValueError("duplicate/missing observations")
    for case in cases:
        row = indexed[case["case_id"]]
        if (
            row["snapshot"]["source"] != digest(case["sources"])
            or row["split"] != case["split"]
        ):
            raise ValueError("frozen source/split drift")
    return [indexed[c["case_id"]] for c in cases]
