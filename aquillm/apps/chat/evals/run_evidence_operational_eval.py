"""Run one predeclared operational mode/profile/repetition through real ASGI."""

import argparse
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from apps.chat.evals.evidence_operational import (
    WORKLOAD_SHA256,
    assess,
    expand,
    load_workload,
)
from apps.chat.evals.evidence_quality_eval import digest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--mode", choices=("preservation", "combined"), required=True)
    parser.add_argument("--profile", choices=("pilot", "one_action"), required=True)
    parser.add_argument("--repetition", choices=(1, 2, 3), type=int, required=True)
    parser.add_argument("--backend", choices=("fixture", "live"), required=True)
    parser.add_argument("--live-manifest", type=Path)
    parser.add_argument("--seed", action="store_true")
    parser.add_argument("--reviews", type=Path)
    parser.add_argument("--observations", type=Path)
    parser.add_argument("--rollout-evidence", type=Path)
    args = parser.parse_args(argv)
    data = load_workload(args.workload)
    workloads = [w for w in data["workloads"] if w["profile"] == args.profile]
    cases = [expand(w, data) for w in workloads]
    args.concurrency, args.cache_state = 1, "cold"
    reviews = (
        json.loads(args.reviews.read_text(encoding="utf-8")) if args.reviews else {}
    )
    original = None
    if args.observations:
        original = json.loads(args.observations.read_text(encoding="utf-8"))
        if any(
            original.get(k) != getattr(args, k)
            for k in ("mode", "backend", "profile", "repetition")
        ):
            parser.error("saved operational run mismatch")
        from apps.chat.evals.evidence_quality_review import load_observations

        rows = load_observations(original, cases, backend=args.backend, mode=args.mode)
        metadata = {k: v for k, v in original.items() if k != "observations"}
    elif args.backend == "live":
        if not args.live_manifest or os.getenv("RAG_EVAL_ISOLATED") != "1":
            parser.error("live requires --live-manifest and RAG_EVAL_ISOLATED=1")
        from apps.chat.evals.evidence_quality_live import run_live

        overrides = data["profiles"][args.profile]["overrides"]
        saved = {k: os.getenv(k) for k in overrides}
        os.environ.update(overrides)
        try:
            rows, metadata = asyncio.run(run_live(cases, args))
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
    else:
        from apps.chat.evals.run_evidence_quality_eval import fixture_observation

        rows = [fixture_observation(c, args.mode) for c in cases]
        metadata = {"runtime_verified": False}
    revision = (
        original["revision"]
        if original
        else subprocess.run(
            ["rtk", "git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )

    results = []
    for workload, row in zip(workloads, rows, strict=True):
        run_id = f"ops-v2/{workload['workload_id']}/{args.mode}/r{args.repetition:02}"
        row.update(run_id=run_id, repetition=args.repetition, profile=args.profile)
        if not original:
            row["code_revision"] = revision
        results.append(
            assess(workload, data, row, reviews.get(run_id, row.get("human_review")))
        )
    report = {
        **metadata,
        "kind": "evidence-operational-report",
        "schema_version": 2,
        "gate_version": "evidence-activation-v2",
        "backend": args.backend,
        "mode": args.mode,
        "profile": args.profile,
        "repetition": args.repetition,
        "corpus_sha256": WORKLOAD_SHA256,
        "workload_version": "ops-v2",
        "planned_run_ids": data["schedule"]["planned_run_ids"],
        "revision": revision,
        "observations": results,
        "activation_eligible": False,
    }
    if args.live_manifest:
        report["manifest_sha256"] = digest(
            json.loads(args.live_manifest.read_text(encoding="utf-8"))
        )
    from apps.chat.evals.evidence_quality_review import dirty_checkout

    report["dirty"] = original.get("dirty", True) if original else dirty_checkout()
    if args.rollout_evidence:
        from apps.chat.evals.evidence_quality_review import attach_evidence

        report.update(
            attach_evidence(
                report, json.loads(args.rollout_evidence.read_text(encoding="utf-8"))
            )
        )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
