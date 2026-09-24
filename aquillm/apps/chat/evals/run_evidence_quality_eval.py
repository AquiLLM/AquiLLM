"""Frozen evidence-quality runner; live mode enters the actual ASGI consumer."""

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from apps.chat.evals.evidence_observation_json import normalize
from apps.chat.evals.evidence_quality_eval import MODES, digest, evaluate, load_cases
from apps.chat.evals.evidence_quality_gates import aggregate, compare


def fixture_observation(case, mode):
    # Plumbing sample only. Gold labels are never fabricated model answers.
    return {
        "mode": mode,
        "backend": "fixture",
        "answer": "",
        "delivered": [],
        "citations": [],
        "snapshot": {
            "source": digest(case["sources"]),
            "answer": "fixture-none",
            "reranker": "fixture-none",
            "hardware": "none",
        },
        "provenance_complete": False,
        "timings_ms": {},
        "stop_reason": "fixture",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path(__file__).with_name("evidence_quality_cases.json"),
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "heldout"), required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--backend", choices=("fixture", "live"), required=True)
    parser.add_argument("--live-manifest", type=Path)
    parser.add_argument("--reviews", type=Path)
    parser.add_argument(
        "--observations",
        type=Path,
        help="Re-score saved observations after human review; no provider calls",
    )
    parser.add_argument("--rollout-evidence", type=Path)
    parser.add_argument("--require-activation", action="store_true")
    parser.add_argument("--compare", type=Path, nargs=4)
    parser.add_argument("--targets", type=Path)
    parser.add_argument("--operational-reports", type=Path, nargs="+", default=[])
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument(
        "--cache-state", choices=("cold", "warm", "unknown"), default="unknown"
    )
    parser.add_argument("--seed", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.concurrency <= 16:
        parser.error("concurrency must be within 1..16")
    cases = [c for c in load_cases(args.cases) if c["split"] == args.split]
    reviews = (
        json.loads(args.reviews.read_text(encoding="utf-8")) if args.reviews else {}
    )
    original = None
    if args.observations:
        from apps.chat.evals.evidence_quality_review import load_observations

        original = json.loads(args.observations.read_text(encoding="utf-8"))
        observations = load_observations(
            original, cases, backend=args.backend, mode=args.mode
        )
        metadata = {
            k: v
            for k, v in original.items()
            if k not in ("observations", "summary", "gate", "activation_eligible")
        }
    elif args.backend == "live":
        if not args.live_manifest or os.getenv("RAG_EVAL_ISOLATED") != "1":
            parser.error("live requires --live-manifest and RAG_EVAL_ISOLATED=1")
        from apps.chat.evals.evidence_quality_live import run_live

        observations, metadata = asyncio.run(run_live(cases, args))
    else:
        observations, metadata = [fixture_observation(c, args.mode) for c in cases], {}
    revision = (
        original["revision"]
        if original
        else subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    if not original:
        for row in observations:
            row["code_revision"] = revision
    results = [
        evaluate(c, o, reviews.get(c["case_id"], o.get("human_review")))
        for c, o in zip(cases, observations, strict=True)
    ]
    report = {
        "schema_version": 1,
        "backend": args.backend,
        "mode": args.mode,
        "split": args.split,
        "revision": revision,
        "corpus_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        "activation_eligible": False,
        "quality_claim": "unmeasured"
        if args.backend == "fixture"
        else "pending_four_arm_human_review",
        "observations": results,
        "summary": aggregate(results),
        **metadata,
    }
    from apps.chat.evals.evidence_quality_review import dirty_checkout

    report["dirty"] = original.get("dirty", True) if original else dirty_checkout()
    if args.rollout_evidence:
        from apps.chat.evals.evidence_quality_review import attach_evidence

        report.update(
            attach_evidence(
                report, json.loads(args.rollout_evidence.read_text(encoding="utf-8"))
            )
        )
    if args.compare:
        reports = [json.loads(p.read_text(encoding="utf-8")) for p in args.compare]
        if len({r["mode"] for r in reports}) != 4:
            parser.error("four distinct report modes required")
        report["gate"] = compare(
            {r["mode"]: r for r in reports},
            json.loads(args.targets.read_text(encoding="utf-8"))
            if args.targets
            else None,
            [
                json.loads(p.read_text(encoding="utf-8"))
                for p in args.operational_reports
            ],
        )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            normalize(report),
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return (
        0
        if not args.require_activation
        or report.get("gate", {}).get("activation_eligible")
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
