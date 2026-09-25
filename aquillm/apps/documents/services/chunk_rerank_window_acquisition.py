"""Acquisition hook; never discovers endpoints or opens a fresh turn ledger."""

from time import monotonic

from .chunk_rerank_results import ScoredRerankResult
from .chunk_rerank_selection_provider import current_selection_scorer


def dispatch_windowed(query, chunks, top_k, turn_budget, scorer=None, *, shadow=False):
    from .chunk_rerank_config import rerank_shadow_scoring, rerank_text_mode

    active = rerank_text_mode() == "windowed"
    if active or (shadow and rerank_shadow_scoring() and turn_budget is not None):
        result = rerank_windowed(
            query, chunks, top_k, turn_budget=turn_budget, scorer=scorer
        )
        if active:
            return result
    return None


def rerank_windowed(query, chunks, top_k, *, turn_budget=None, scorer=None):
    rows = tuple(chunks)
    deadline = monotonic() + (turn_budget.remaining_ms() / 1000 if turn_budget else 0)
    scorer = scorer or current_selection_scorer(
        deadline=deadline, turn_budget=turn_budget, windowed=True
    )
    if scorer.budget is not turn_budget:
        raise ValueError("acquisition must share the turn ledger")
    result = scorer.score_windows(query, rows, phase="acquisition")
    order = {row.pk: index for index, row in enumerate(rows)}
    ranked = tuple(
        s.chunk_pk
        for s in sorted(result.scores, key=lambda s: (-s.value, order[s.chunk_pk]))
    )
    return ScoredRerankResult((ranked or tuple(order))[:top_k], result)
