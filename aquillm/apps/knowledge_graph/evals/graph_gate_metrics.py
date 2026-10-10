"""Measured identity counts and paired, overlapping graph retrieval cohorts."""

from itertools import combinations
from math import log2

from .graph_gate_observations import IDENTITY_STRATA, RETRIEVAL_STRATA
from .retrieval_eval_statistics import paired_graph_interval, wilson_95


def identity_metrics(rows, policy, reasons):
    automatic = [r for r in rows if r["automatic"]]
    candidates = [r for r in rows if r["candidate"]]
    correct = sum(r["expected_same"] for r in automatic)
    total = len(automatic)
    precision = correct / total if total else None
    interval = wilson_95(correct, total)
    if total < policy.min_automatic_pairs:
        reasons.append("automatic_pairs_insufficient")
    if precision is None or precision < policy.min_automatic_precision:
        reasons.append("automatic_precision_below_floor")
    if (
        interval["lower"] is None
        or interval["lower"] < policy.min_automatic_precision_lower_95
    ):
        reasons.append("automatic_precision_interval_below_floor")
    strata = {}
    for name in IDENTITY_STRATA:
        cohort = [r for r in automatic if name in r["strata"]]
        good = sum(r["expected_same"] for r in cohort)
        strata[name] = dict(
            total=len(cohort), correct=good, interval=wilson_95(good, len(cohort))
        )
        if len(cohort) < policy.min_pairs_per_stratum:
            reasons.append(f"identity_stratum_insufficient:{name}")
        if cohort and good / len(cohort) < policy.min_automatic_precision:
            reasons.append(f"identity_stratum_precision_below_floor:{name}")
    conflicts = [r for r in rows if r["known_conflict"]]
    merges = sum(r["automatic"] for r in conflicts)
    if not conflicts or merges:
        reasons.append("known_conflict_merges_or_unknown")
    return dict(
        automatic_correct=correct,
        automatic_incorrect=total - correct,
        automatic_total=total,
        automatic_precision=precision,
        interval=interval,
        candidate_correct=sum(r["expected_same"] for r in candidates),
        candidate_total=len(candidates),
        candidate_precision=(
            sum(r["expected_same"] for r in candidates) / len(candidates)
            if candidates
            else None
        ),
        known_conflict_total=len(conflicts),
        known_conflict_merges=merges,
        strata=strata,
    )


def retrieval_scores(row, arm):
    observation, labels = row[arm], row["labels"]
    top, relevant = observation["ranked"][:10], set(labels["relevant"])
    hits = sum(x in relevant for x in top)
    ideal = sum(1 / log2(i + 2) for i in range(min(10, len(relevant))))
    dcg = sum(1 / log2(i + 2) for i, item in enumerate(top) if item in relevant)
    no_answer = observation["abstained"] and not observation["ranked"]
    return dict(
        recall_at_10=hits / len(relevant) if relevant else float(no_answer),
        ndcg_at_10=dcg / ideal if ideal else float(no_answer),
        precision_at_10=hits / len(top) if top else float(labels["no_answer"]),
        no_answer_correct=float(no_answer == labels["no_answer"]),
        coverage_label_accuracy=float(
            observation["coverage"] == labels["coverage"]
            and observation["exhaustive"] is False
        ),
    )


def retrieval_metrics(rows, policy, intended, reasons):
    if len(rows) < policy.min_query_pairs:
        reasons.append("query_pairs_insufficient")
    if not any(r["labels"]["no_answer"] for r in rows):
        reasons.append("no_answer_observations_missing")
    scores = {
        r["query_id"]: {a: retrieval_scores(r, a) for a in ("baseline", "candidate")}
        for r in rows
    }
    for row in rows:
        for arm in ("baseline", "candidate"):
            obs = row[arm]
            if obs["authorization_violations"] or obs["citation_violations"]:
                reasons.append("authorization_or_citation_violation")
            if obs["ranked"] != obs["repeated_ranked"]:
                reasons.append("nondeterministic_ranking")
            if obs["completion"] != "completed":
                reasons.append("completion_failed_or_unknown")
            if obs["latency_ms"] > obs["latency_budget_ms"]:
                reasons.append("latency_budget_exceeded")
        if (
            row["baseline"]["latency_budget_ms"]
            != row["candidate"]["latency_budget_ms"]
        ):
            reasons.append("unmatched_latency_budget")
        candidate = scores[row["query_id"]]["candidate"]
        if candidate["no_answer_correct"] != 1:
            reasons.append("no_answer_failed")
        if candidate["coverage_label_accuracy"] != 1:
            reasons.append("coverage_label_failed")
    cohorts = {"aggregate": rows}
    for size in range(1, len(RETRIEVAL_STRATA) + 1):
        for names in combinations(RETRIEVAL_STRATA, size):
            cohort = [r for r in rows if set(names) <= set(r["strata"])]
            if size == 1 or cohort:
                cohorts["&".join(names)] = cohort
                if size == 1 and len(cohort) < policy.min_pairs_per_stratum:
                    reasons.append(f"retrieval_stratum_insufficient:{names[0]}")
    intervals = {}
    for name, cohort in cohorts.items():
        metrics = {}
        for metric in (
            "recall_at_10",
            "ndcg_at_10",
            "precision_at_10",
            "no_answer_correct",
            "coverage_label_accuracy",
        ):
            delta = [
                scores[r["query_id"]]["candidate"][metric]
                - scores[r["query_id"]]["baseline"][metric]
                for r in cohort
            ]
            interval = paired_graph_interval(delta)
            metrics[metric] = interval
            if (
                not delta
                or interval["mean"] < -policy.max_metric_decline
                or interval["lower"] < -policy.max_metric_decline
            ):
                reasons.append(
                    "aggregate_retrieval_regression"
                    if name == "aggregate"
                    else f"retrieval_cohort_regression:{name}:{metric}"
                )
            if (
                name == intended
                and metric in ("recall_at_10", "ndcg_at_10")
                and (not delta or interval["lower"] <= 0)
            ):
                reasons.append(f"intended_improvement_unproven:{name}:{metric}")
        intervals[name] = metrics
    return dict(query_pairs=len(rows), cohorts=intervals, observations=scores)
