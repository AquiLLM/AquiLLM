"""Admit complete pointwise source plans without widening the final pair cap."""

from apps.documents.services.chunk_rerank_window_adapter import WindowSelectionScorer
from lib.evidence_observation import publish


def admit_window_sources(items, *, scorer, query, reused):
    """Stable ranked first-fit; transport admission still owns every real lease."""
    if (
        not isinstance(scorer, WindowSelectionScorer)
        or scorer.scoring_kind != "pointwise"
        or getattr(scorer.provider, "scoring_kind", "pointwise") != "pointwise"
        or scorer.budget is None
    ):
        return items
    costs = []
    for item in items:
        plan = scorer.plan(query, item.chunk)
        if plan.preparation_coverage != "complete" or not plan.required_window_ids:
            return items
        cached = item.chunk.pk in reused or scorer.cached_score(query, item.chunk)
        costs.append(0 if cached else len(plan.required_window_ids))
    remaining = max(
        0, scorer.budget.limits.final_pairs - scorer.budget.pairs_used["final"]
    )
    admitted, available = [], remaining
    for item, cost in zip(items, costs):
        if cost <= available:
            admitted.append(item)
            available -= cost
    if len(admitted) != len(items):
        publish("selection_window_admission", {
            "candidate_count": len(items), "admitted_count": len(admitted),
            "planned_new_pairs_upper_bound": sum(costs),
            "admitted_new_pairs_upper_bound": remaining - available,
            "remaining_final_pairs": remaining,
        })
    return tuple(admitted)
