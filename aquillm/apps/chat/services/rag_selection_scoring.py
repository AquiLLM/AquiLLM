"""Build one authorized pool with comparable model or explicitly ranked relevance."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from copy import copy
from time import monotonic

from apps.chat.services.rag_retrieval import FusedRetrievalPool
from apps.chat.services.rag_selection_hydration import hydrate_pool_rows
from apps.chat.services.rag_selection_inputs import prepare_final_input
from apps.chat.services.rag_selection_relevance import selection_relevance
from apps.chat.services.rag_selection_reuse import _reused_scores
from apps.chat.services.rag_selection_types import PreparedSelection, SelectionCandidate
from apps.chat.services.rag_selection_window_admission import (
    admit_window_sources,
)
from apps.collections.services.retrieval_authorization import (
    RetrievalAuthorizationContext,
)
from apps.documents.services.chunk_rerank_config import rerank_score_concurrency
from apps.documents.services.chunk_rerank_results import (
    PassageScore,
    fingerprint_text,
    score_identity_for_chunk,
    validate_score_set,
)
from apps.documents.services.chunk_rerank_scoring import (
    SelectionScorer,
    score_missing_pairs,
)
from apps.documents.services.chunk_rerank_window_adapter import (
    WindowSelectionScorer,
    expected_score_fingerprint,
)
from apps.documents.services.source_deadline import (
    bound_source_finalization,
    check_source_deadline,
    source_preparation_scope,
)


@bound_source_finalization
def prepare_selection_candidates(
    *,
    pool: FusedRetrievalPool,
    primary_query: str,
    authorization: RetrievalAuthorizationContext,
    deadline: float,
    allow_new_scores: bool,
    scorer: SelectionScorer | None = None,
    chunk_loader=None,
    clock: Callable[[], float] = monotonic,
    turn_budget=None,
    source_mode=False,
    token_ceiling=3500,
    source_windows=None,
) -> PreparedSelection:
    """Never widen pool membership or mix model and rank-derived scales."""
    if len(pool.rows) > 45:
        raise ValueError("candidate union exceeds the hard cap")
    if turn_budget is not None:
        deadline = min(
            deadline, clock() + turn_budget.scoring_remaining_ms("final") / 1000
        )

    def hydrate():
        if source_mode:
            from apps.chat.services.rag_source_hydration import hydrate_source_rows

            return hydrate_source_rows(
                pool.rows,
                authorization,
                question=primary_query,
                token_ceiling=token_ceiling,
                turn_budget=turn_budget,
                source_windows=source_windows,
                chunk_loader=chunk_loader,
            )
        return hydrate_pool_rows(pool.rows, authorization, chunk_loader=chunk_loader)

    with source_preparation_scope(deadline, clock) if source_mode else nullcontext():
        first = hydrate()
    initial_by_id = {
        item.chunk.pk: (item.source_fingerprint, item.excerpt) for item in first
    }
    used_scorer = scorer
    if used_scorer is None and (allow_new_scores or pool.source_score_sets):
        from apps.documents.services.chunk_rerank_selection_provider import (
            current_selection_scorer,
        )

        used_scorer = current_selection_scorer(
            deadline=deadline, clock=clock, turn_budget=turn_budget
        )
    if isinstance(used_scorer, WindowSelectionScorer):
        used_scorer.deadline = min(used_scorer.deadline, deadline)
    reused: dict[int, PassageScore] = {}
    new_scores: dict[int, PassageScore] = {}
    new_pairs = 0
    score_start = clock()
    fallback_reason = None
    final_inputs = {}
    admitted = first
    if used_scorer is None:
        fallback_reason = "scorer_unavailable"
    elif first:
        try:
            if not isinstance(used_scorer, WindowSelectionScorer):
                prepare_emitted = getattr(used_scorer, "prepare_emitted_pair", None)
                if not callable(prepare_emitted):
                    raise ValueError("scorer cannot prepare exact final evidence")
                for item in first:
                    source_text = (
                        item.prepared_evidence.source.text
                        if item.prepared_evidence is not None
                        else item.chunk.content
                    )
                    final_input = prepare_final_input(
                        chunk_id=item.chunk.pk,
                        source_text=source_text,
                        emitted_text=item.excerpt,
                        query=primary_query,
                        prepare_pair=prepare_emitted,
                    )
                    if (
                        final_input is None
                        or final_input.source_fingerprint != item.source_fingerprint
                    ):
                        raise ValueError("prepared final evidence is incompatible")
                    final_inputs[item.chunk.pk] = final_input
            reused = _reused_scores(
                pool=pool,
                hydrated=first,
                query=primary_query,
                scorer=used_scorer,
                final_inputs=final_inputs if final_inputs else None,
            )
            if allow_new_scores and clock() < deadline:
                admitted = admit_window_sources(
                    first, scorer=used_scorer, query=primary_query, reused=reused,
                )
            missing = tuple(item for item in admitted if item.chunk.pk not in reused)
            if missing and allow_new_scores and clock() < deadline:
                # Listwise scores depend on the complete ordered pool.
                to_score = (
                    first if (used_scorer.scoring_kind == "listwise") else missing
                )
                prepared_pairs = None
                if final_inputs:
                    ordered_inputs = tuple(
                        final_inputs[item.chunk.pk] for item in to_score
                    )
                    if tuple(item.chunk_id for item in ordered_inputs) != tuple(
                        item.chunk.pk for item in to_score
                    ):
                        raise ValueError("prepared final score order changed")
                    prepared_pairs = tuple(item.pair for item in ordered_inputs)
                scoring_chunks = []
                for item in to_score:
                    chunk = item.chunk
                    if final_inputs and item.prepared_evidence is not None:
                        chunk = copy(chunk)
                        chunk.content = item.prepared_evidence.source.text
                    scoring_chunks.append(chunk)
                submitted: list[int] = []
                fresh = score_missing_pairs(
                    query=primary_query,
                    chunks=tuple(scoring_chunks),
                    scorer=used_scorer,
                    deadline=deadline,
                    max_inflight=min(6, rerank_score_concurrency()),
                    clock=clock,
                    on_submit=lambda count: submitted.append(count),
                    turn_budget=turn_budget,
                    prepared_pairs=prepared_pairs,
                )
                new_pairs = sum(submitted)
                if fresh.status == "complete" and (
                    fresh.scorer_fingerprint == used_scorer.scorer_fingerprint
                    and fresh.query_fingerprint == fingerprint_text(primary_query)
                    and fresh.scoring_kind == used_scorer.scoring_kind
                ):
                    if fresh.candidate_order != tuple(
                        item.chunk.pk for item in to_score
                    ):
                        raise ValueError("new score pool changed")
                    validate_score_set(
                        fresh,
                        authorized_identities=tuple(
                            (
                                item.chunk.pk,
                                item.chunk.doc_id,
                                item.chunk.chunk_number,
                                final_inputs[item.chunk.pk].source_fingerprint,
                                final_inputs[item.chunk.pk].pair_fingerprint,
                            )
                            if final_inputs
                            else score_identity_for_chunk(
                                item.chunk,
                                effective_pair_fingerprint=expected_score_fingerprint(
                                    used_scorer, primary_query, item.chunk
                                ),
                            )
                            for item in to_score
                        ),
                        expected_query_fingerprint=fingerprint_text(primary_query),
                        expected_scorer_fingerprint=used_scorer.scorer_fingerprint,
                    )
                    new_scores = {score.chunk_pk: score for score in fresh.scores}
            if missing and not new_scores and not allow_new_scores:
                fallback_reason = "new_scores_disabled"
        except (TypeError, ValueError):
            fallback_reason = "score_incompatible"
    scoring_duration_ms = max(0.0, (clock() - score_start) * 1000.0)

    # Permissions and content may change while inference is running.
    from apps.chat.services.rag_source_hydration import revalidate_prepared_rows

    if source_mode:
        check_source_deadline()
    current = (
        revalidate_prepared_rows(first, authorization, chunk_loader=chunk_loader)
        if source_mode
        else hydrate()
    )
    retained = tuple(
        item
        for item in current
        if initial_by_id.get(item.chunk.pk) == (item.source_fingerprint, item.excerpt)
    )
    scores = {**reused, **new_scores}
    relevance, status, fallback_reason, published_scores = selection_relevance(
        pool=pool, retained=retained, admitted=admitted, scores=scores,
        final_inputs=final_inputs, scorer=used_scorer,
        live=clock() < deadline and (turn_budget is None or turn_budget.can_publish()),
        allow_partial=len(admitted) < len(first), fallback_reason=fallback_reason,
    )
    if status == "rank_fallback":
        fallback_reason = fallback_reason or (
            "deadline" if clock() >= deadline else "incomplete_scores"
        )
    rank_by_id = {
        row.get("chunk_id", row.get("i")): rank
        for rank, row in enumerate(pool.rows, start=1)
    }
    from apps.chat.services.rag_source_hydration import source_candidate_cost

    candidates = tuple(
        SelectionCandidate(
            item.chunk.pk,
            str(item.chunk.doc_id),
            item.chunk.chunk_number,
            item.excerpt,
            value,
            rank_by_id[item.chunk.pk],
            item.source_fingerprint,
            item.row,
            getattr(item, "prepared_evidence", None),
            source_candidate_cost(item, turn_budget) if source_mode else None,
            published_scores[item.chunk.pk].value
            if item.chunk.pk in published_scores else None,
            used_scorer.scorer_fingerprint
            if item.chunk.pk in published_scores else None,
            used_scorer.scoring_kind
            if item.chunk.pk in published_scores else "rank_only",
            "uncalibrated" if item.chunk.pk in published_scores else "unavailable",
        )
        for item, value in zip(retained, relevance)
    )
    return PreparedSelection(
        candidates,
        status,
        len(reused),
        new_pairs,
        scoring_duration_ms,
        fallback_reason,
    )


__all__ = ["PreparedSelection", "prepare_selection_candidates"]
