"""Private zero-inference route; exactly one unchanged native finalizer owns rows."""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter
from typing import TYPE_CHECKING

from apps.documents.models import TextChunk

from .chunk_rerank_results import RerankScoreSet
from .chunk_search_candidates import HybridCandidateSnapshot
from .chunk_search_legacy_graph import graph_diagnostics
from .chunk_search_revalidation import reconcile_candidate_order
from .chunk_search_scope import validate_requested_authorization
from .closed_search_finalizer import _finalize_closed_ranking
from .native_search_capture import capture_search, hydrated_rows
from .native_search_schedule import (
    NATIVE_CLOSURE_RESERVE_MS,
    NATIVE_CLOSURE_SCHEDULE_VERSION,
)
from .representation_contracts import SourceScopeUnavailable
from .source_loading import bounded_source_database

if TYPE_CHECKING:
    from .chunk_search import CandidateRankingResult


@dataclass(frozen=True, slots=True)
class _NativeSearchClosureResult:
    snapshot: HybridCandidateSnapshot
    ranking: CandidateRankingResult
    final_rows: tuple[tuple[TextChunk, ...], tuple[TextChunk, ...], list[TextChunk]]
    score_set: RerankScoreSet | None
    graph_diagnostics: dict[str, object]
    hybrid_pool: tuple[TextChunk, ...]
    preclosure_pool_count: int
    materialized_union_status: str = "unavailable"
    acquisition_schedule: str = "native_closure"
    schedule_version: str = NATIVE_CLOSURE_SCHEDULE_VERSION


def _try_native_search_closure(
    snapshot,
    query,
    top_k,
    *,
    authorize,
    dependencies,
    overlay_enabled,
    observation=None,
):
    """Only initial ineligibility returns None; captured failures never switch routes.

    The optional diagnostics dictionary records effective scored fallback without
    carrying authority. This private entry has no production dispatch yet.
    """
    if observation is not None:
        observation["acquisition_schedule"] = "scored"
    captured = capture_search(
        snapshot, query, top_k, authorize, dependencies, overlay_enabled
    )
    if captured is None:
        return None
    started = perf_counter()

    def diagnostics(status, count=0):
        return graph_diagnostics(
            started_at=started,
            seed_count=len(snapshot.graph_seeds),
            candidate_count=count,
            status=status,
            algorithm_signature=None,
            version_signature=None,
        )

    attempt = None
    if overlay_enabled:
        from apps.knowledge_graph.retrieval.closed_graph_attempt_v2 import (
            _capture_closed_graph_attempt,
        )

        attempt = _capture_closed_graph_attempt(
            snapshot,
            query,
            authorize.authorization,
            dependencies,
            diagnostics=diagnostics,
        )
        if attempt is None:
            return None  # Still before any optional work or route consumption.
    captured.check()
    pool, graph_observation = captured.baseline, {}
    if attempt is not None:
        outcome = attempt.run_native_closure(
            closure_reserve_ms=NATIVE_CLOSURE_RESERVE_MS
        )
        captured.check()  # Admission.check now runs in the restored source context.
        if outcome.kind == "invalid":
            raise SourceScopeUnavailable("native_search_original_scope_unavailable")
        graph_observation = outcome.diagnostics
        if outcome.kind == "success":
            pool = outcome.pool
        else:
            captured.admission.counts.update(
                fallback_top_up_skipped=True, fallback_top_up_count=0
            )
            graph_observation["source_admission"] = captured.admission.diagnostics()
        snapshot = captured.admission.refresh_snapshot(snapshot, pool)
    else:
        with captured.header[0][5]:
            captured.check()
            with bounded_source_database(captured.scope.runtime, captured.scope.using):
                captured.check()
                validate_requested_authorization(
                    captured.scope.runtime, authorize.authorization, snapshot.documents
                )
                captured.check()
            captured.check()
        captured.check()

    from .chunk_search import CandidateRankingResult

    if not hydrated_rows(pool):
        raise SourceScopeUnavailable("native_search_invalid_candidate_pool")
    combined = reconcile_candidate_order(pool, scope=captured.scope)
    if len(combined) != len(pool) or {id(row) for row in combined} != {
        id(row) for row in pool
    }:
        raise SourceScopeUnavailable("native_search_invalid_candidate_pool")
    ranking = CandidateRankingResult(
        combined_candidates=combined,
        graph_candidates=(),
        ranked_results=combined[:top_k],
        inaccessible_candidate_count=0,
        materialization_ms=0.0,
        rerank_ms=0.0,
        score_set=None,
        rerank_status="unavailable",
    )
    captured.check()
    final = _finalize_closed_ranking(
        snapshot, ranking, authorize=authorize, query=query, top_k=top_k
    )
    captured.check()  # Includes DB cleanup and pure completion reentrancy.
    if final is None:
        raise SourceScopeUnavailable("native_search_finalizer_unavailable")
    snapshot, final_rows, score_set = final
    ranked = tuple(final_rows[2])
    result = _NativeSearchClosureResult(
        snapshot,
        replace(ranking, combined_candidates=ranked, ranked_results=ranked),
        final_rows,
        score_set,
        graph_observation,
        ranked,
        len(combined),
    )
    captured.check()
    if observation is not None:
        observation["acquisition_schedule"] = "native_closure"
    return result
