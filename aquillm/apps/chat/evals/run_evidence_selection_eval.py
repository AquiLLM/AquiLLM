"""Offline replay of real selector decisions on identical labeled candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

from apps.chat.evals.evidence_selection_fixtures import load_cases
from apps.chat.evals.evidence_selection_metrics import (
    aggregate,
    paired_bootstrap_interval,
    selection_metrics,
)
from apps.chat.services.rag_legacy_selection import diversify_evidence_chunks
from apps.chat.services.rag_selection import select_evidence
from apps.chat.services.rag_selection_policy import choose_selection_profile
from apps.chat.services.rag_selection_types import SelectionProfile

POLICIES = ("legacy", "relevance_only", "fixed_mmr", "adaptive")
DEFAULT_CASES = Path(__file__).with_name("evidence_selection_cases.yaml")


def _legacy(case):
    """Reproduce existing two round-robin passes and early passage cutoff."""
    ordered = sorted(case.candidates, key=lambda c: (c.fused_rank, c.chunk_id))
    rows = diversify_evidence_chunks(
        [dict(c.row) for c in ordered], case.limits.max_per_document
    )
    rows = rows[: case.limits.max_passages]
    rows = diversify_evidence_chunks(rows, case.limits.max_per_document)
    by_id = {c.chunk_id: c for c in ordered}
    selected, used = [], 0
    for row in rows:
        candidate = by_id[row["chunk_id"]]
        cost = max(1, (len(candidate.text) + 3) // 4)
        if used + cost <= case.limits.token_budget:
            selected.append(candidate)
            used += cost
    return tuple(selected), used


def evaluate_case(case, policy):
    if policy not in POLICIES:
        raise ValueError("unknown selection policy")
    started = perf_counter()
    if policy == "legacy":
        selected, tokens = _legacy(case)
        profile_name = "legacy"
    else:
        profile = choose_selection_profile(case.question)
        if case.profile:
            cues = {"focused": "Who?", "balanced": "Explain", "breadth": "Compare"}
            profile = choose_selection_profile(cues[case.profile])
        if policy == "relevance_only":
            profile = SelectionProfile("relevance_only", 1.0, 1.0, "replay-v1")
        elif policy == "fixed_mmr":
            profile = SelectionProfile("fixed_mmr", 0.9, 0.1, "replay-v1")
        result = select_evidence(
            case.candidates,
            profile=profile,
            limits=case.limits,
            score_status=case.score_status,
        )
        selected, tokens, profile_name = (
            result.candidates,
            result.estimated_tokens,
            profile.name,
        )
    selection_ms = (perf_counter() - started) * 1000
    selected_ids = [c.chunk_id for c in selected]
    expected = case.labels["expected_adaptive"]
    available = {c.chunk_id for c in case.candidates}
    return {
        "id": case.identifier,
        "family": case.family,
        "policy": policy,
        "profile": profile_name,
        "score_status": case.score_status,
        "selected_chunk_ids": selected_ids,
        "selected_citations": [c.row["citation"] for c in selected],
        "metrics": selection_metrics(case, selected, tokens),
        "missing_upstream_gold": sorted(
            set(case.labels["required_chunks"]) - available
        ),
        "expectation_passed": selected_ids == list(expected)
        if policy == "adaptive" and expected
        else None,
        "selection_ms": selection_ms,
        "scoring_observations": {
            key: case.scoring_observations.get(key)
            for key in ("reused_pairs", "new_pairs", "stage_ms")
        },
    }


def evaluate(cases, policy, *, manifest_digest):
    rows = [evaluate_case(case, policy) for case in cases]
    baseline = [evaluate_case(case, "legacy") for case in cases]
    comparisons = {}
    for key in (
        "ndcg",
        "supporting_chunk_recall",
        "required_pair_recall",
        "aspect_coverage",
    ):
        differences = [
            row["metrics"][key] - base["metrics"][key]
            for row, base in zip(rows, baseline, strict=True)
            if row["metrics"][key] is not None and base["metrics"][key] is not None
        ]
        comparisons[key] = paired_bootstrap_interval(differences)
    return {
        "schema_version": 1,
        "policy": policy,
        "manifest_digest": manifest_digest,
        "quality_gate_status": "unmeasured_real_corpus",
        "cases": rows,
        "aggregate": aggregate(rows),
        "families": {
            family: aggregate([row for row in rows if row["family"] == family])
            for family in sorted({row["family"] for row in rows})
        },
        "comparison_to_legacy": comparisons,
        "measurement_limits": [
            "Synthetic cases verify mechanics, not corpus quality.",
            "Missing scoring observations remain unmeasured.",
            "Latency is local replay selection time, not model or serving latency.",
            "Heuristic repetition uses attenuated lexical overlap >=0.8.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--policy", choices=POLICIES, default="adaptive")
    parser.add_argument(
        "--split",
        choices=("regression", "development", "held_out"),
        default="regression",
    )
    args = parser.parse_args(argv)
    cases = tuple(case for case in load_cases(args.cases) if case.split == args.split)
    if not cases:
        parser.error("no cases in requested split")
    report = evaluate(
        cases,
        args.policy,
        manifest_digest=hashlib.sha256(args.cases.read_bytes()).hexdigest(),
    )
    payload = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return int(any(row["expectation_passed"] is False for row in report["cases"]))


if __name__ == "__main__":
    raise SystemExit(main())
