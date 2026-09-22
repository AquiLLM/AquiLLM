"""Identity-based replay metrics and paired bootstrap confidence intervals."""

from __future__ import annotations

import math
import random
from statistics import mean

from apps.chat.services.rag_selection_similarity import snippet_redundancy


def selection_metrics(case, selected, estimated_tokens):
    ids = {c.chunk_id for c in selected}
    grades = case.labels["grades"]
    ideal = sorted(grades.values(), reverse=True)[: case.limits.max_passages]

    def dcg(values):
        return sum(
            (2**grade - 1) / math.log2(rank + 2) for rank, grade in enumerate(values)
        )

    denominator = dcg(ideal)
    required = set(case.labels["required_chunks"])
    pairs = case.labels["required_pairs"]
    aspects = case.labels["aspects"]
    repeated = sum(
        any(snippet_redundancy(c, old) >= 0.8 for old in selected[:i])
        for i, c in enumerate(selected)
    )
    groups = case.labels["redundancy_groups"]
    return {
        "ndcg": dcg([grades.get(c.chunk_id, 0) for c in selected]) / denominator
        if denominator
        else None,
        "supporting_chunk_recall": len(ids & required) / len(required)
        if required
        else None,
        "required_pair_recall": sum(set(pair) <= ids for pair in pairs) / len(pairs)
        if pairs
        else None,
        "aspect_coverage": sum(bool(ids & set(group)) for group in aspects.values())
        / len(aspects)
        if aspects
        else None,
        "heuristic_repetition_rate": repeated / len(selected) if selected else 0.0,
        "human_redundant_group_rate": sum(len(ids & set(group)) > 1 for group in groups)
        / len(groups)
        if groups
        else None,
        "document_count": len({c.doc_id for c in selected}),
        "estimated_tokens": estimated_tokens,
        "forbidden_selected": len(ids & set(case.labels["forbidden_chunks"])),
    }


def aggregate(rows):
    keys = {key for row in rows for key in row["metrics"]}
    result = {}
    for key in sorted(keys):
        values = [
            row["metrics"][key] for row in rows if row["metrics"].get(key) is not None
        ]
        result[key] = mean(values) if values else None
    return result


def paired_bootstrap_interval(deltas, *, resamples=2000):
    """Percentile 95% interval on paired per-question differences, fixed RNG."""
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
