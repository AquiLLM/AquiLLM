"""Independently labeled support rates and predeclared cohort uncertainty."""

import math
from statistics import NormalDist

from .evidence_quality_eval import MODES
from .evidence_review_subject import validated_review

DIMENSIONS = (
    "quote_accuracy",
    "source_entailment",
    "inference",
    "numerical_conditions",
    "contradiction_free",
    "coverage",
)
LABELS = (
    *DIMENSIONS,
    "false_complete",
    "unsupported_answer",
    "abstained",
    "abstention_correct",
)


def support_labels(row):
    review = validated_review(row, row.get("human_review"))
    labels = review.get("support_labels") if review else None
    return (
        labels
        if isinstance(labels, dict) and all(type(labels.get(k)) is bool for k in LABELS)
        else None
    )


def publication_matches(row):
    return (
        isinstance(row.get("answer"), str)
        and bool(row.get("final_message_id"))
        and row.get("publication_capture_complete") is True
        and all(
            row.get(k) == row["answer"]
            for k in ("final_stream", "final_delta", "persisted_answer")
        )
    )


def wilson(successes, count, confidence):
    if not count:
        return {"count": 0, "rate": None, "lower": None, "upper": None}
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    p, denom = successes / count, 1 + z * z / count
    center = (p + z * z / (2 * count)) / denom
    half = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denom
    return {
        "count": count,
        "rate": p,
        "lower": max(0, center - half),
        "upper": min(1, center + half),
    }


def measure_cohorts(rows, criteria, splits=("development", "heldout")):
    """No pooled average may conceal a missing or failing domain/depth/split."""
    modes = criteria["modes"]
    if (
        not isinstance(modes, list)
        or not modes
        or len(set(modes)) != len(modes)
        or not set(modes) <= set(MODES)
    ):
        raise ValueError("missing arm policy")
    if any(r["mode"] not in modes for r in rows):
        raise ValueError("undeclared support arm")
    return {
        mode: _measure_arm([r for r in rows if r["mode"] == mode], criteria, splits)
        for mode in modes
    }


def _measure_arm(rows, criteria, splits):
    rates = (
        "max_false_complete",
        "max_unsupported_answer",
        "min_abstention_quality",
        "min_answer_coverage",
    )
    for key in (*rates, "confidence"):
        value = criteria[key]
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError("invalid numeric support criterion")
    if not 0 < criteria["confidence"] < 1 or criteria["method"] != "wilson":
        raise ValueError("unsupported uncertainty method")
    minimum = criteria["minimum_samples"]
    domains, depths = criteria["domains"], criteria["depths"]
    if (
        type(minimum) is not int
        or minimum < 1
        or not isinstance(domains, list)
        or not domains
        or any(not isinstance(x, str) or not x for x in domains)
        or len(set(domains)) != len(domains)
        or set(depths) != {"routine", "deeper"}
    ):
        raise ValueError("missing cohort policy")
    measured = {}
    for split in splits:
        for domain in domains:
            for depth in depths:
                group = [
                    r
                    for r in rows
                    if (r["split"], r["domain"], r["depth"]) == (split, domain, depth)
                ]
                labels = [support_labels(r) for r in group]
                if len(group) < minimum or any(v is None for v in labels):
                    raise ValueError("missing independently labeled support samples")

                def interval(values):
                    return wilson(sum(values), len(values), criteria["confidence"])

                metrics = {
                    key: interval([v[key] for v in labels])
                    for key in ("false_complete", "unsupported_answer")
                }
                metrics["abstention_quality"] = interval(
                    [v["abstention_correct"] for v in labels if v["abstained"]]
                )
                metrics["answer_coverage"] = interval(
                    [not v["abstained"] for v in labels]
                )
                metrics["dimensions"] = {
                    k: interval([v[k] for v in labels]) for k in DIMENSIONS
                }
                if (
                    any(v["false_complete"] for v in labels)
                    or any(
                        not v[k]
                        for v in labels
                        if not v["abstained"]
                        for k in DIMENSIONS
                    )
                    or any(
                        metrics[k]["upper"] > criteria["max_" + k]
                        for k in ("false_complete", "unsupported_answer")
                    )
                    or any(
                        metrics[k]["lower"] is None
                        or metrics[k]["lower"] < criteria["min_" + k]
                        for k in ("abstention_quality", "answer_coverage")
                    )
                ):
                    raise ValueError("support acceptance or uncertainty failed")
                measured[f"{split}/{domain}/{depth}"] = metrics
    if any(
        r["domain"] not in domains
        or r["depth"] not in depths
        or r["split"] not in ("development", "heldout")
        for r in rows
    ):
        raise ValueError("undeclared support cohort")
    return measured
