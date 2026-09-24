"""A complete schedule cannot substitute one_action for the pilot limit subtype."""

from apps.chat.evals.evidence_operational import (
    WORKLOAD_SHA256,
    expand,
    load_workload,
    validate_attachment,
)
from apps.chat.evals.evidence_quality_eval import digest


def test_complete_attachment_requires_pilot_limit_proof(monkeypatch):
    data = load_workload()
    snapshot = {k: k for k in ("answer", "reranker", "hardware", "embedding")}
    quality = {
        mode: {
            "revision": "r",
            "observations": [
                {"snapshot": snapshot, "comparison_controls": {"max_actions": 3}}
            ],
        }
        for mode in ("preservation", "combined")
    }
    reports = []
    for run_id in data["schedule"]["planned_run_ids"]:
        _, case_id, mode, repetition = run_id.split("/")
        workload = next(w for w in data["workloads"] if w["workload_id"] == case_id)
        profile = workload["profile"]
        row = {
            "run_id": run_id,
            "code_revision": "r",
            "case_id": case_id,
            "mode": mode,
            "events": [],
            "trace_sha256": digest([]),
            "profile": profile,
            "repetition": int(repetition[1:]),
            "snapshot": {
                **snapshot,
                "source": digest(expand(workload, data)["sources"]),
            },
            "comparison_controls": {"max_actions": 1 if profile == "one_action" else 3},
            "operational": {
                "refinement_pass": True,
                "limit_pass": profile == "one_action",
            },
        }
        reports.append(
            {
                "backend": "live",
                "dirty": False,
                "corpus_sha256": WORKLOAD_SHA256,
                "planned_run_ids": data["schedule"]["planned_run_ids"],
                "manifest_sha256": "m",
                "concurrency": 1,
                "pair_capability_verified": True,
                "revision": "r",
                "mode": mode,
                "profile": profile,
                "repetition": int(repetition[1:]),
                "observations": [row],
            }
        )
    monkeypatch.setattr(
        "apps.chat.evals.evidence_quality_review.verified_flags",
        lambda _: {"runtime_verified": True},
    )
    # Scope: subtype joining only. Real trace/partial predicates have separate tests.
    monkeypatch.setattr(
        "apps.chat.evals.evidence_operational.assess",
        lambda w, d, r, h: {
            "actual_safety": {"passed": True},
            "operational": r["operational"],
        },
    )
    result = validate_attachment(reports, quality)
    assert result["blocking_reasons"] == [
        "combined: live pair_or_deadline unproven",
        "preservation: live pair_or_deadline unproven",
    ]
    for report in reports:
        if report["profile"] == "pilot":
            report["observations"][0]["operational"]["limit_pass"] = True
    assert validate_attachment(reports, quality)["passed"]
