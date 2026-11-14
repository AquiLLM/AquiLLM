"""Choose bounded work ordering without changing source ranking or provenance."""

from apps.documents.services.chunk_rerank_window_adapter import WindowSelectionScorer

from .rag_preparation_order import take_preparation_order
from .rag_recovery_origins import take_recovery_order


def prepare_ordered_selection(prepare, **kwargs):
    order = take_preparation_order(kwargs["pool"], kwargs["primary_query"], kwargs["authorization"])
    neutral = order is not None
    if order is None:
        order = take_recovery_order(kwargs["pool"], kwargs["primary_query"], kwargs["authorization"])
    if order is None:
        return None
    scorer = kwargs["scorer"]
    if scorer is None and (kwargs["allow_new_scores"] or kwargs["pool"].source_score_sets):
        from apps.documents.services.chunk_rerank_selection_provider import current_selection_scorer

        scorer = current_selection_scorer(deadline=kwargs["deadline"], clock=kwargs["clock"],
                                          turn_budget=kwargs["turn_budget"])
    pointwise = (isinstance(scorer, WindowSelectionScorer) and scorer.scoring_kind == "pointwise"
                 and getattr(scorer.provider, "scoring_kind", "pointwise") == "pointwise")
    if not neutral and not pointwise:
        return None  # Preserve the established scoped-only compatibility path.
    from .rag_recovery_preparation import prepare_recovery_selection

    preflight = kwargs.pop("recovery_preflight")
    return prepare_recovery_selection(order, prepare=prepare, **{**kwargs, "scorer": scorer},
                                      answer_preflight=preflight, allow_deferred=pointwise)
