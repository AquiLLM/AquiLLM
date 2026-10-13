"""Descriptive physical recall/latency only; failures never become samples."""
import math
import statistics
from collections import defaultdict


def distribution(values):
    if not values:
        return None
    values = sorted(values)
    return {"n": len(values), "median_ms": statistics.median(values),
            "p95_ms": values[math.ceil(.95 * len(values)) - 1],
            "min_ms": values[0], "max_ms": values[-1]}


def summarize(records, planned):
    grouped, pairs, comparisons = defaultdict(list), defaultdict(dict), []
    for row in records:
        if row["kind"] != "timed":
            continue
        if row["status"] == "observed":
            grouped[(row["scope"], row["arm"])].append(row["rank_ms"])
        pairs[(row["scope"], row["query"], row["repeat"])][row["arm"]] = row
    for (scope, query, repeat), arms in pairs.items():
        exact, candidate = arms.get("production_exact"), arms.get("experimental_direct_distance")
        result = {"scope": scope, "query": query, "repeat": repeat, "status": "inconclusive"}
        if exact and candidate and exact["status"] == candidate["status"] == "observed":
            baseline = exact["ranked"]
            if len(baseline) != min(45, scope):
                result["reason"] = "exact_underfill"
            elif any(baseline[i]["distance"] == baseline[i + 1]["distance"]
                     for i in range(min(10, len(baseline) - 1))):
                result["reason"] = "unexpected_distance_tie"
            else:
                result.update(status="observed", recall={}, raw_hydrated_recall={})
                for k in (1, 5, 10):
                    n = min(k, scope)
                    target = {r["chunk_id"] for r in baseline[:n]}
                    actual = {r["chunk_id"] for r in candidate["ranked"][:n]}
                    result["recall"][str(k)] = len(target & actual) / n
                    raw = {r["chunk_id"] for r in candidate["observer"]["raw_hydrated_candidates"][:n]}
                    result["raw_hydrated_recall"][str(k)] = len(target & raw) / n
                result["paired_rank_ms_difference"] = candidate["rank_ms"] - exact["rank_ms"]
                result["candidate_underfill"] = max(0, min(45, scope) - len(candidate["ranked"]))
        comparisons.append(result)
    return {"decision": "inconclusive", "activation": "off", "semantic_quality": "unmeasured",
        "planned_invocations": planned, "retained_invocations": len(records),
        "unattempted": planned - len(records),
        "failed_or_unavailable": sum(r["status"] != "observed" for r in records),
        "latencies": [{"scope": scope, "arm": arm, **distribution(values)}
                      for (scope, arm), values in sorted(grouped.items())],
        "paired_comparisons": comparisons,
        "uncertainty": "descriptive individual paired differences; no independent-sample CI",
        "cache": "uncontrolled and mixed; no cold-cache claim",
        "ann": "inspect raw plans; direct-distance arm name does not prove an index plan"}
