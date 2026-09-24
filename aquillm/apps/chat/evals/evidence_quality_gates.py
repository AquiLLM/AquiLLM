"""Conservative four-arm quality/responsiveness decisions, never fixture activation."""

import math
from collections import Counter
from statistics import mean

from .evidence_quality_comparison import (
    frozen_comparison_errors,
    p95_ratio_interval,
    paired_bootstrap_interval,
)
from .evidence_quality_eval import MODES, join_arms
from .evidence_quality_review import verified_flags


def percentile(values, fraction):
    if not values or any(
        not isinstance(x, (float, int)) or not math.isfinite(x) for x in values
    ):
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def aggregate(rows):
    normal = [r for r in rows if r["quality_aggregation"] == "normal"]
    result = {}
    for key in ("support_recall", "faithfulness", "qualification_accuracy", "ndcg"):
        values = [r[key] for r in normal if r.get(key) is not None]
        result[key] = {
            "mean": mean(values) if values else None,
            "count": len(values),
            "unknown": len(normal) - len(values),
        }
    result["cohorts"] = {}
    for depth in ("routine", "deeper"):
        for state in ("cold", "warm"):
            cohort = [
                r
                for r in normal
                if r["depth"] == depth and r.get("cache_state") == state
            ]
            stages = {k for r in cohort for k in r.get("timings_ms", {})}
            result["cohorts"][f"{depth}/{state}"] = {
                stage: {
                    "p50": percentile(
                        [r.get("timings_ms", {}).get(stage) for r in cohort], 0.5
                    ),
                    "p95": percentile(
                        [r.get("timings_ms", {}).get(stage) for r in cohort], 0.95
                    ),
                    "count": len(cohort),
                }
                for stage in stages
            }
    windows = [
        w
        for r in rows
        for w in r.get("window_scores", [])
        if w.get("score") is not None
    ]
    result["window_count_bias"] = {
        str(n): {"count": len(group), "mean_score": mean(w["score"] for w in group)}
        for n in sorted({w["windows"] for w in windows})
        if (group := [w for w in windows if w["windows"] == n])
    }
    known = [r for r in rows if r.get("rank_fallback") is not None]
    result["rank_fallback_rate"] = (
        mean(r["rank_fallback"] for r in known) if known else None
    )
    result["safety_unknown"] = sum(
        r.get("safety") is None
        for r in rows
        if r["quality_aggregation"] == "safety_only"
    )
    result["coverage_counts"] = dict(Counter(str(r.get("coverage")) for r in rows))
    result["stop_reason_counts"] = dict(
        Counter(str(r.get("stop_reason")) for r in rows)
    )
    result["dispatch_counts"] = dict(
        Counter(d["kind"] for r in rows for d in r.get("dispatches", []))
    )
    result["pair_cost"] = {
        phase: {"total": sum(values), "observed_turns": len(values)}
        for phase in ("acquisition", "final")
        if (values := [r["pairs"][phase] for r in rows if "pairs" in r])
    }
    return result


def compare(reports, targets=None, operational_reports=()):
    """Missing observations or uncertainty block activation; no equivalence claim."""
    groups = join_arms({m: reports[m]["observations"] for m in MODES})
    reasons, comparisons = [], {}
    reasons.extend(frozen_comparison_errors(reports))
    from .evidence_review_subject import quality_review_errors

    reasons.extend(quality_review_errors(reports))
    from .evidence_operational import validate_attachment

    operational = validate_attachment(operational_reports, reports)
    reasons.extend(operational["blocking_reasons"])
    for mode, report in reports.items():
        verified = verified_flags(report)
        if report.get("dirty") is not False:
            reasons.append(f"{mode}: unverified clean code revision")
        if (
            mode in ("preservation", "combined")
            and report.get("pair_capability_verified") is not True
        ):
            reasons.append(f"{mode}: unverified serving-worker pair capability")
        if report.get("backend") != "live":
            reasons.append(f"{mode}: fixture is ineligible")
        for requirement in (
            "runtime_verified",
            "corpus_human_reviewed",
            "deterministic_regressions_passed",
            "concurrent_load_verified",
            "cold_warm_verified",
            "completion_reserve_measured",
        ):
            if (report if requirement == "concurrent_load_verified" else verified).get(
                requirement
            ) is not True:
                reasons.append(f"{mode}: missing {requirement}")
        for row in report["observations"]:
            safety_only = row["quality_aggregation"] == "safety_only"
            candidate = mode in ("preservation", "combined")
            if (not safety_only or candidate) and (
                row.get("citation_violations") or row.get("authorization_violations")
            ):
                reasons.append(f"{mode}: authorization/citation violation")
            if (
                row.get("provenance_complete") is not True
                or row.get("dispatch_accounting_complete") is not True
            ):
                reasons.append(f"{mode}: incomplete runtime observation")
            if candidate and row.get("actual_safety", {}).get("passed") is not True:
                reasons.append(f"{mode}: unproven safety outcome")
    normal = [g for g in groups if g["baseline"]["quality_aggregation"] == "normal"]
    for mode in MODES[1:]:
        metrics = {}
        for key in ("support_recall", "faithfulness", "qualification_accuracy", "ndcg"):
            pairs = [(g["baseline"].get(key), g[mode].get(key)) for g in normal]
            applicable = [(a, b) for a, b in pairs if a is not None and b is not None]
            metrics[key] = paired_bootstrap_interval([b - a for a, b in applicable])
            if key != "ndcg" and len(applicable) != len(pairs):
                reasons.append(f"{mode}: unknown {key}")
            interval = metrics[key]
            if interval["mean"] is not None and (
                interval["mean"] < 0 or interval["lower"] < 0
            ):
                reasons.append(f"{mode}: {key} decline or inconclusive interval")
        if mode in ("preservation", "combined"):
            targeted = [
                g
                for g in normal
                if g["baseline"]["scenario"]
                in (
                    "tail",
                    "negation_boundary",
                    "four_passages",
                    "plural_followup",
                    "ordinal_citation",
                    "second_search",
                )
            ]
            diffs = [
                g[mode]["support_recall"] - g["baseline"]["support_recall"]
                for g in targeted
                if g[mode].get("support_recall") is not None
                and g["baseline"].get("support_recall") is not None
            ]
            metrics["targeted_support"] = paired_bootstrap_interval(diffs)
            if (
                not diffs
                or len(diffs) != len(targeted)
                or metrics["targeted_support"]["lower"] <= 0
            ):
                reasons.append(f"{mode}: targeted improvement unproven")
        for depth in ("routine", "deeper"):
            cohort = [g for g in normal if g["baseline"]["depth"] == depth]
            for quality in ("support_recall", "faithfulness", "qualification_accuracy"):
                deltas = [
                    g[mode][quality] - g["baseline"][quality]
                    for g in cohort
                    if g[mode].get(quality) is not None
                    and g["baseline"].get(quality) is not None
                ]
                interval = paired_bootstrap_interval(deltas)
                metrics[f"{depth}/{quality}"] = interval
                if len(deltas) != len(cohort) or not deltas or interval["lower"] < 0:
                    reasons.append(f"{mode}: {depth}/{quality} decline or unknown")
            for stage in ("first_grounded", "completion"):
                times = {
                    arm: [g[arm].get("timings_ms", {}).get(stage) for g in cohort]
                    for arm in ("baseline", mode)
                }
                base, active = (
                    percentile(times[arm], 0.95) for arm in ("baseline", mode)
                )
                target = (targets or {}).get(depth, {}).get(stage)
                uncertainty = p95_ratio_interval(times["baseline"], times[mode])
                metrics[f"{depth}/{stage}/p95_ratio"] = uncertainty
                if base is None or active is None or not target:
                    reasons.append(
                        f"{mode}: missing {depth}/{stage} measurement or numeric target"
                    )
                elif active > target or active > base * (
                    1 if depth == "routine" else 1.2
                ):
                    reasons.append(f"{mode}: {depth}/{stage} responsiveness failed")
                elif uncertainty["upper"] > (1 if depth == "routine" else 1.2):
                    reasons.append(
                        f"{mode}: {depth}/{stage} responsiveness inconclusive"
                    )
        comparisons[mode] = metrics
    return {
        "activation_eligible": not reasons,
        "gate_version": "evidence-activation-v2",
        "operational_evidence": operational,
        "blocking_reasons": sorted(set(reasons)),
        "paired_comparisons": comparisons,
        "targets_ms": targets,
    }
