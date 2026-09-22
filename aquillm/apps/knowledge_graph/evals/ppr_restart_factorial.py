"""Four-arm synthetic evidence replay; does not claim answer-model quality."""

from copy import deepcopy

from apps.chat.evals.evidence_selection_fixtures import parse_case
from apps.chat.evals.evidence_selection_metrics import aggregate
from apps.chat.evals.run_evidence_selection_eval import evaluate_case


def factorial_replay(cases, graph_results, *, baseline_policy="fixed_020"):
    arms = {name: [] for name in ("current", "A_only", "B_only", "A_and_B")}
    for case in cases:
        evidence = case.raw.get("evidence_replay")
        if evidence is None:
            continue
        for arm, a_enabled, b_enabled in (
            ("current", False, False),
            ("A_only", True, False),
            ("B_only", False, True),
            ("A_and_B", True, True),
        ):
            graph_policy = "adaptive_v1" if b_enabled else baseline_policy
            graph = graph_results[(case.raw["id"], graph_policy)]
            allowed = set(graph["candidate_chunk_ids"]) | set(
                evidence.get("baseline_ids", [])
            )
            raw = deepcopy(evidence)
            raw.update(
                id=case.raw["id"],
                split=case.raw["split"],
                question=case.raw["question"],
            )
            raw["candidates"] = [
                row for row in raw["candidates"] if row["chunk_id"] in allowed
            ]
            raw.pop("expected_adaptive", None)
            ranked = sorted(
                raw["candidates"], key=lambda row: (-row["relevance"], row["chunk_id"])
            )
            for rank, row in enumerate(ranked, 1):
                row["fused_rank"] = rank
            raw["candidates"] = ranked
            arms[arm].append(
                evaluate_case(parse_case(raw), "adaptive" if a_enabled else "legacy")
            )
    return {
        "baseline_policy": baseline_policy,
        "status": "synthetic_evidence_replay" if any(arms.values()) else "unmeasured",
        "arms": {
            name: {"cases": rows, "aggregate": aggregate(rows)}
            for name, rows in arms.items()
        },
        "limitations": "Uses frozen relevance labels and graph snapshots; "
        "live retrieval, answer synthesis, and provider latency are unmeasured.",
    }


def quality_strata(rows):
    """Keep seed size and policy abstention visible beside support status."""
    result = {}
    for name, field in (
        ("seed_count", "seed_count"),
        ("policy_reason", "reason"),
        ("intent", "intent"),
    ):
        result[name] = {}
        for value in sorted({str(row[field]) for row in rows}):
            group = [row for row in rows if str(row[field]) == value]
            result[name][value] = {"count": len(group), "metrics": aggregate(group)}
    return result
