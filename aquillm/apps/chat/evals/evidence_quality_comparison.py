"""Frozen case completeness and paired uncertainty for tail latency."""

import math
import random
from pathlib import Path
from statistics import mean

from .evidence_quality_eval import FROZEN_SHA256, digest, load_cases


def paired_bootstrap_interval(deltas, *, resamples=2000):
    """Same fixed-seed paired estimator as selection, without app imports."""
    if type(resamples) is not int or resamples < 1:
        raise ValueError("invalid resample count")
    if not deltas:
        return {"count": 0, "mean": None, "lower": None, "upper": None}
    if any(not math.isfinite(value) for value in deltas):
        raise ValueError("nonfinite paired differences")
    rng = random.Random(20260922)
    samples = sorted(mean(rng.choices(deltas, k=len(deltas))) for _ in range(resamples))
    return {
        "count": len(deltas),
        "mean": mean(deltas),
        "lower": samples[int(0.025 * (resamples - 1))],
        "upper": samples[int(0.975 * (resamples - 1))],
    }


def frozen_comparison_errors(reports):
    cases = load_cases(Path(__file__).with_name("evidence_quality_cases.json"))
    reasons = []
    indexed = {c["case_id"]: c for c in cases}
    conditions = set()
    for mode, report in reports.items():
        split = report.get("split")
        expected = {c["case_id"] for c in cases if c["split"] == split}
        rows = report.get("observations", [])
        if (
            report.get("corpus_sha256") != FROZEN_SHA256
            or not expected
            or len(rows) != len(expected)
            or {r["case_id"] for r in rows} != expected
        ):
            reasons.append(f"{mode}: incomplete frozen split")
        for row in rows:
            case = indexed.get(row["case_id"])
            if case and (
                row.get("snapshot", {}).get("source") != digest(case["sources"])
                or any(
                    row.get(k) != case[k]
                    for k in ("depth", "scenario", "quality_aggregation", "split")
                )
            ):
                reasons.append(f"{mode}: frozen source/cohort drift")
        conditions.add((report.get("concurrency"), report.get("cache_state"), split))
    if len(conditions) != 1 or any(
        c is None or c < 2 or state not in ("cold", "warm")
        for c, state, _ in conditions
    ):
        reasons.append("unmatched concurrent/cache conditions")
    return reasons


def p95_ratio_interval(baseline, candidate, *, samples=2000):
    if (
        not baseline
        or len(baseline) != len(candidate)
        or any(
            type(x) not in (int, float) or not math.isfinite(x) or x <= 0
            for x in baseline + candidate
        )
    ):
        return {"lower": None, "upper": None, "ratio": None, "count": 0}

    def tail(values):
        return sorted(values)[math.ceil(len(values) * 0.95) - 1]

    rng = random.Random(20260922)
    ratios = []
    for _ in range(samples):
        indices = [rng.randrange(len(baseline)) for _ in baseline]
        ratios.append(
            tail([candidate[i] for i in indices]) / tail([baseline[i] for i in indices])
        )
    ratios.sort()
    return {
        "lower": ratios[int(samples * 0.025)],
        "upper": ratios[int(samples * 0.975)],
        "ratio": tail(candidate) / tail(baseline),
        "count": len(baseline),
    }
