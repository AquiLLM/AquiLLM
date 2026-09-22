"""Versioned offline PPR controls on identical authorized synthetic snapshots."""

from __future__ import annotations

import argparse
import json
import tracemalloc
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from statistics import mean
from time import perf_counter

from apps.chat.evals.evidence_selection_metrics import paired_bootstrap_interval
from apps.knowledge_graph.evals.ppr_restart_factorial import (
    factorial_replay,
    quality_strata,
)
from apps.knowledge_graph.evals.ppr_restart_fixtures import load_cases
from apps.knowledge_graph.evals.ppr_restart_reference import reference_run
from apps.knowledge_graph.retrieval.ppr import canonical_algorithm_json
from apps.knowledge_graph.retrieval.ppr_kernel import run_ppr_kernel
from apps.knowledge_graph.retrieval.ppr_policy import ppr_policy_input_digest
from apps.knowledge_graph.retrieval.production_runtime_support import graph_candidates
from apps.knowledge_graph.retrieval.projected_ppr import (
    ppr_projected_v1,
    prepare_projected_ppr_inputs,
)
from apps.knowledge_graph.retrieval.projected_ppr_execution import (
    execute_adaptive_projected_ppr,
)
from apps.knowledge_graph.retrieval.projected_types import projected_snapshot_checksum
from apps.knowledge_graph.retrieval.topology.contracts import projected_seed_checksum

FIXED = {"fixed_015": 0.15, "fixed_020": 0.20, "fixed_035": 0.35, "fixed_050": 0.50}
POLICIES = (*FIXED, "adaptive_v1")
DEFAULT_CASES = Path(__file__).with_name("ppr_restart_cases.json")


def evaluate_case(case, policy):
    if policy not in POLICIES:
        raise ValueError("unsupported restart policy")
    snapshot, config, seeds = case.snapshot, case.config, case.seeds
    owned_trace = not tracemalloc.is_tracing()
    if owned_trace:
        tracemalloc.start()
    started = perf_counter()
    try:
        if policy == "adaptive_v1":
            result = execute_adaptive_projected_ppr(
                snapshot=snapshot,
                seeds=seeds,
                base_config=config,
                signals=case.signals,
                expected_branch=case.signals.branch_kind,
                deadline_check=lambda: None,
            )
            ranking, restart, reason = (
                result.ranking,
                result.decision.restart,
                result.decision.reason,
            )
            signature, execution = (
                result.algorithm_signature,
                result.execution_signature,
            )
            policy_digest = result.policy_input_digest
        else:
            restart, reason = FIXED[policy], "fixed_control"
            effective = replace(config, ppr_restart=restart)
            if policy == "fixed_020":
                ranking = ppr_projected_v1(
                    snapshot=snapshot, seeds=seeds, config=config
                )
            else:
                ranking = run_ppr_kernel(
                    nodes=snapshot.identity_keys,
                    edges=prepare_projected_ppr_inputs(
                        snapshot=snapshot, seeds=seeds, config=config
                    ),
                    seeds={seed.identity_key: seed.mass for seed in seeds},
                    config=effective,
                    order_key=lambda value: value,
                )
            signature = sha256(
                b"ppr_restart_eval_fixed_v1\0" + canonical_algorithm_json(effective)
            ).hexdigest()
            policy_digest = ppr_policy_input_digest(case.signals)
            execution = sha256(
                (
                    signature
                    + projected_snapshot_checksum(snapshot)
                    + projected_seed_checksum(seeds)
                    + policy_digest
                ).encode()
            ).hexdigest()
        candidates = graph_candidates(
            snapshot=snapshot,
            identity_scores=ranking.scores,
            maximum=config.max_candidates,
        )
        stage_ms = (perf_counter() - started) * 1000
        peak = tracemalloc.get_traced_memory()[1] if owned_trace else None
    finally:
        if owned_trace:
            tracemalloc.stop()
    ids = [case.chunk_ids[row.chunk_key] for row in candidates]
    edges = prepare_projected_ppr_inputs(snapshot=snapshot, seeds=seeds, config=config)
    first, converged, reference = reference_run(
        snapshot=snapshot,
        config=config,
        edges=edges,
        seeds=seeds,
        restart=restart,
    )
    reference_candidates = graph_candidates(
        snapshot=snapshot,
        identity_scores=tuple(converged.items()),
        maximum=config.max_candidates,
    )
    reference_ids = [case.chunk_ids[row.chunk_key] for row in reference_candidates]
    reference["candidate_order_changed"] = ids != reference_ids
    reference["candidate_chunk_ids"] = reference_ids
    required, irrelevant = (
        set(case.raw.get("required_chunks", [])),
        set(case.raw.get("irrelevant_chunks", [])),
    )
    groups = case.raw.get("required_groups", [])
    return {
        "id": case.raw["id"],
        "policy": policy,
        "intent": case.signals.intent,
        "branch": case.signals.branch_kind.value,
        "support_status": case.signals.support_status,
        "support_summary": case.raw.get("support_summary", {}),
        "effective_restart": restart,
        "effective_iterations": 8,
        "effective_config": json.loads(
            canonical_algorithm_json(replace(config, ppr_restart=restart))
        ),
        "policy_version": "ppr_restart_policy_v1"
        if policy == "adaptive_v1"
        else "fixed_control_v1",
        "reason": reason,
        "abstained": policy == "adaptive_v1" and restart == 0.2,
        "snapshot_checksum": projected_snapshot_checksum(snapshot),
        "seed_checksum": projected_seed_checksum(seeds),
        "seed_count": len(seeds),
        "algorithm_signature": signature,
        "execution_signature": execution,
        "policy_input_digest": policy_digest,
        "candidate_chunk_ids": ids,
        "reference": reference,
        "one_step_scores": {case.node_names[node]: value for node, value in first},
        "eight_step_scores": {
            case.node_names[node]: value for node, value in ranking.scores
        },
        "stage_with_allocation_tracing_ms": stage_ms,
        "python_peak_bytes": peak,
        "metrics": {
            "candidate_recall": len(required & set(ids)) / len(required)
            if required
            else None,
            "support_group_recall": sum(set(group) <= set(ids) for group in groups)
            / len(groups)
            if groups
            else None,
            "irrelevant_expansion": len(irrelevant & set(ids)) / len(ids)
            if ids
            else 0.0,
        },
    }


def _mean_metric(rows, key):
    values = [row["metrics"][key] for row in rows if row["metrics"][key] is not None]
    return mean(values) if values else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--policy", choices=POLICIES, default="adaptive_v1")
    parser.add_argument(
        "--split",
        choices=("regression", "development", "held_out"),
        default="regression",
    )
    parser.add_argument(
        "--fixed-choice",
        choices=tuple(FIXED),
        help="Fixed winner frozen on development data",
    )
    args = parser.parse_args(argv)
    cases = tuple(
        case for case in load_cases(args.cases) if case.raw["split"] == args.split
    )
    if not cases:
        parser.error("no cases in selected split")
    results = {
        (case.raw["id"], policy): evaluate_case(case, policy)
        for case in cases
        for policy in POLICIES
    }
    controls = {
        policy: _mean_metric(
            [results[(case.raw["id"], policy)] for case in cases], "candidate_recall"
        )
        for policy in FIXED
    }
    best = (
        max(
            (p for p in FIXED if controls[p] is not None),
            key=lambda p: controls[p],
            default=None,
        )
        if args.split == "development"
        else args.fixed_choice
    )
    baseline = best or "fixed_020"
    rows = [results[(case.raw["id"], args.policy)] for case in cases]
    deltas = [
        row["metrics"]["candidate_recall"]
        - results[(row["id"], baseline)]["metrics"]["candidate_recall"]
        for row in rows
        if row["metrics"]["candidate_recall"] is not None
    ]
    report = {
        "schema_version": 1,
        "policy": args.policy,
        "cases": rows,
        "manifest_digest": sha256(args.cases.read_bytes()).hexdigest(),
        "quality_gate_status": "unmeasured_real_corpus",
        "best_fixed_policy": best,
        "fixed_controls_candidate_recall": controls,
        "paired_comparison": {
            "baseline": baseline,
            **paired_bootstrap_interval(deltas),
        },
        "abstention_rate": mean(row["abstained"] for row in rows),
        "strata": {
            status: {
                "count": sum(row["support_status"] == status for row in rows),
                "candidate_recall": _mean_metric(
                    [row for row in rows if row["support_status"] == status],
                    "candidate_recall",
                ),
            }
            for status in ("supported", "insufficient", "unknown")
        },
        "factorial_replay": factorial_replay(cases, results),
        "best_fixed_factorial_replay": (
            factorial_replay(cases, results, baseline_policy=best) if best else None
        ),
        "quality_strata": quality_strata(rows),
        "limitations": (
            "Synthetic fixtures do not establish held-out quality. "
            "Timings include Python allocation tracing; native/server memory "
            "and live answer quality are unmeasured."
        ),
    }
    payload = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
