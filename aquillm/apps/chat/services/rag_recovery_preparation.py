"""Finish ordinary source work before bounded optional recovery preparation."""

from dataclasses import replace

from apps.documents.services.native_search_schedule import NATIVE_CLOSURE_RESERVE_MS
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.services.source_deadline import source_preparation_scope
from apps.documents.services.source_deadline import _DEADLINE
from apps.documents.services.source_loading import SourcePreparationLimited
from lib.evidence_observation import publish
from lib.llm.evidence_guard import ContextLimited
from lib.retrieval.operation import _OPERATION

from .rag_retrieval import FusedRetrievalPool
from .rag_preparation_order import PreparationOrder
from .rag_recovery_origins import capture_operation_check
from .rag_recovery_request import UnsupportedRecoveryRequest
from .rag_selection_policy import normalize_relevance_scores
from .rag_selection_types import OmittedRecoverySource, PreparedSelection
from .rag_source_hydration import revalidate_source_candidates


def prepare_recovery_selection(order, *, prepare, pool, primary_query, authorization,
        deadline, allow_new_scores, scorer, chunk_loader, clock, turn_budget,
        source_mode, token_ceiling, source_windows, answer_preflight=None, allow_deferred=True):
    """Two existing pointwise preparations share one final lease and scorer cache.

    The second tier is optional as a batch. Partial or unprepared optional rows
    never become selectable rank-only placeholders.
    """
    parent_scope, operation = _DEADLINE.get(), _OPERATION.get()
    operation_identity, operation_check = capture_operation_check(operation, turn_budget)
    start = clock()
    operation_identity()
    pairs_before = turn_budget.pairs_used["final"]
    capability = (scorer.provider, scorer.pair_counter, scorer.scorer_fingerprint,
                  scorer.scoring_kind, scorer.budget, scorer.rerank_state) if allow_deferred else None
    by_id = {row.get("chunk_id", row.get("i")): row for row in pool.rows}
    identities = {identity[0]: identity for identity in order.identities}
    neutral = type(order) is PreparationOrder
    ordinary_ids = order.initial_ids if neutral else order.ordinary_ids
    optional_ids = order.deferred_ids if neutral else order.optional_ids
    ordinary_scores = order.fallback_scores if neutral else order.ordinary_scores
    limited_reason = "preparation_deferred" if neutral else "optional_recovery_pruned"
    full_ranks = {pk: rank for rank, pk in enumerate(by_id, 1)}
    fallback_ranks = {pk: rank for rank, pk in enumerate(order.fallback_ids if neutral else ordinary_ids, 1)}

    def run(ids, until, scores):
        return prepare(
            pool=FusedRetrievalPool(tuple(by_id[pk] for pk in ids), pool.source_score_sets, scores),
            primary_query=primary_query, authorization=authorization, deadline=until,
            allow_new_scores=allow_new_scores, scorer=scorer, chunk_loader=chunk_loader,
            clock=clock, turn_budget=turn_budget, source_mode=source_mode,
            token_ceiling=token_ceiling, source_windows=source_windows,
        )

    def parent():
        operation_identity()
        order.check()
        operation_check()
        if _DEADLINE.get() is not parent_scope:
            raise SourceScopeUnavailable("recovery_original_scope_unavailable")
        if parent_scope is not None:
            remaining = int((parent_scope[0] - parent_scope[1]()) * 1000)
            operation_identity()
            if parent_scope[2] is not None:
                if parent_scope[2] is not turn_budget:
                    raise SourceScopeUnavailable("recovery_original_scope_unavailable")
                with turn_budget._lock:
                    remaining = min(remaining, int((turn_budget._deadline - turn_budget._clock()) * 1000))
                operation_identity()
            if remaining <= 0:
                raise SourcePreparationLimited("source preparation deadline")
        with turn_budget._lock:
            scoring_live = capability is None or not capability[-1]._failed
        now = clock()
        order.origins.check_owner()
        if capability is not None:
            current = (scorer.provider, scorer.pair_counter, scorer.scorer_fingerprint,
                       scorer.scoring_kind, scorer.budget, scorer.rerank_state)
            if (any(current[i] is not capability[i] for i in (0,1,4,5))
                    or current[2:4] != capability[2:4]):
                raise SourceScopeUnavailable("recovery_scoring_capability_changed")
        operation_identity()
        if _DEADLINE.get() is not parent_scope:
            raise SourceScopeUnavailable("recovery_original_scope_unavailable")
        if now >= deadline:
            raise SourcePreparationLimited("final source preparation deadline")
        if not scoring_live or (capability is not None and capability[-1]._failed):
            raise SourcePreparationLimited("source scorer unavailable")

    def omitted(ids, reason):
        if neutral:
            reason = "preparation_" + reason.removeprefix("optional_")
        return tuple(OmittedRecoverySource(*identities[pk], reason) for pk in ids)

    order.check()
    operation_check()
    if _DEADLINE.get() is not parent_scope:
        raise SourceScopeUnavailable("recovery_original_scope_unavailable")
    ordinary = run(ordinary_ids, deadline, ordinary_scores)
    if neutral:
        ordinary = replace(ordinary, candidates=tuple(replace(candidate, fused_rank=fallback_ranks[candidate.chunk_id])
                           for candidate in ordinary.candidates))
    if ordinary.score_status != "model" and not neutral:
        return replace(ordinary, acquired_count=len(pool.rows),
                       omitted_optional=omitted(optional_ids, "ordinary_incomplete"))
    parent()
    baseline = replace(ordinary, score_status="partial_model_ranked" if ordinary.score_status == "model" else ordinary.score_status,
        fallback_reason=limited_reason, acquired_count=len(pool.rows),
        omitted_optional=omitted(optional_ids, "answer_preflight_unavailable"))
    if answer_preflight is None:
        return baseline
    try:
        baseline = replace(baseline, prepaid_selection=answer_preflight(baseline, pin=True))
    except UnsupportedRecoveryRequest:
        parent()
        return baseline
    parent()
    if not allow_deferred or ordinary.score_status != "model":
        return replace(baseline, omitted_optional=omitted(optional_ids, "scorer_ineligible"))
    optional = None
    reason = "optional_time_reserve"
    child_deadline = deadline - NATIVE_CLOSURE_RESERVE_MS / 1000
    optional_time = clock() < child_deadline
    operation_identity()
    order.origins.check_owner()
    if optional_time:
        try:
            with source_preparation_scope(child_deadline, clock, turn_budget):
                optional = run(optional_ids, child_deadline, pool.fused_scores)
        except ContextLimited as exc:
            if type(exc) is not ContextLimited or str(exc) != "request_tokenization_limit":
                raise
            reason = "optional_text_limit"
        except SourcePreparationLimited as exc:
            if (type(exc) is not SourcePreparationLimited
                    or str(exc) != "source preparation deadline" or clock() < child_deadline):
                raise
            reason = "optional_child_deadline"
    parent()
    # A provider outage invalidates its publication latch. Do not turn that
    # unrelated failure into a successful recovery of previously cached scores.
    accepted = () if optional is None else tuple(
        candidate for candidate in optional.candidates
        if candidate.scoring_kind == "pointwise" and candidate.raw_score is not None)
    if optional is not None:
        reason = "optional_incomplete_scores"
    candidates = (*ordinary.candidates, *accepted)
    for candidate in candidates:
        if identities[candidate.chunk_id] != (candidate.chunk_id, candidate.doc_id,
                candidate.chunk_number, candidate.source_fingerprint):
            raise SourceScopeUnavailable("recovery_source_changed")
    # No evidence preparation or charged candidate-cost calculation is repeated.
    candidates = revalidate_source_candidates(candidates, authorization, chunk_loader=chunk_loader)
    parent()
    kept = {candidate.chunk_id for candidate in candidates}
    if any(candidate.chunk_id not in kept for candidate in baseline.prepaid_selection.candidates):
        raise SourceScopeUnavailable("recovery_source_changed")
    omissions = omitted((pk for pk in optional_ids if pk not in kept), reason)
    relevance = normalize_relevance_scores(tuple(candidate.raw_score for candidate in candidates))
    # Failed expansion has no vote in ordinary fallback or constant-score ties.
    # A scored combined pool retains its existing mixed-pool tie-break ranks.
    optional_retained = any(pk in kept for pk in optional_ids)
    scoped_retained = any(pk in kept for pk in order.scoped_only_ids) if neutral else optional_retained
    ranks = full_ranks if scoped_retained else fallback_ranks
    candidates = tuple(replace(candidate, relevance=value, fused_rank=ranks[candidate.chunk_id])
                       for candidate, value in zip(candidates, relevance))
    status = "partial_model_ranked" if omissions else "model"
    result = PreparedSelection(candidates, status,
        ordinary.reused_pairs + (optional.reused_pairs if optional else 0),
        turn_budget.pairs_used["final"] - pairs_before,
        max(0, (clock() - start) * 1000),
        limited_reason if omissions else None,
        acquired_count=len(pool.rows), omitted_optional=omissions)
    if optional_retained:
        try:
            result = replace(result, prepaid_selection=answer_preflight(result, pin=False))
        except UnsupportedRecoveryRequest:
            reason = "optional_request_unsupported"
            optional_retained = False
        except ContextLimited as exc:
            if type(exc) is not ContextLimited or str(exc) not in {
                    "request_tokenization_limit", "request_context_exhausted"}:
                raise
            reason = "optional_answer_work_limit"
            optional_retained = False
    if not optional_retained:
        result = replace(baseline,
            candidates=tuple(candidate for candidate in ordinary.candidates if candidate.chunk_id in kept),
            new_pairs=turn_budget.pairs_used["final"] - pairs_before,
            scoring_duration_ms=max(0, (clock() - start) * 1000),
            omitted_optional=omitted(optional_ids, reason))
    parent()
    publish("selection_native_preparation" if neutral else "selection_optional_recovery", {
        "acquired_count": len(pool.rows), "ordinary_count": len(ordinary_ids),
        "retained_count": len(result.candidates), "scored_count": len(result.candidates),
        "omitted_count": len(result.omitted_optional), "status": result.score_status,
    })
    parent()
    return result
