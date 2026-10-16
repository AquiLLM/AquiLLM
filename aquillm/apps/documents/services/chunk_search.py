"""Hybrid chunk retrieval with an optional fail-open graph overlay."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING
from uuid import UUID

import structlog
from django.apps import apps
from django.conf import settings as django_settings

from apps.documents.services import chunk_search_legacy_graph as _legacy_graph
from apps.documents.services.chunk_rerank import (
    _STRICT_EVALUATION_RERANK,
    _fallback_rerank,
    _strict_local_rerank_chunks,
    rerank_chunks,
    rerank_chunks_scored,
)
from apps.documents.services.chunk_rerank_results import RerankScoreSet
from apps.documents.services.chunk_search_authorization import (
    authorized_search_rows,
    finalize_ranking,  # noqa: F401 - retained dynamic scored-flow patch point
)
from apps.documents.services.chunk_search_candidates import (
    _exact_term_query as _candidate_exact_term_query,
)
from apps.documents.services.chunk_search_candidates import (
    _salient_exact_terms as _candidate_salient_exact_terms,
)
from apps.documents.services.chunk_search_candidates import (
    collect_hybrid_candidate_snapshot,
    freeze_authorized_document_scope,
)
from apps.documents.services.chunk_search_logging import (
    completed_search_diagnostics,
    log_search_failure,
)
from apps.documents.services.chunk_search_validation import (
    candidate_identifier as _candidate_identifier,
)
from apps.documents.services.chunk_search_validation import (
    candidate_identity as _candidate_identity,
)
from apps.documents.services.chunk_search_validation import (
    validate_candidate_request,
)
from apps.documents.services.chunk_source_materialization import materialize_graph_rows
from apps.documents.services.hybrid_graph_authorization import (
    HybridGraphRetrievalDependencies,
    is_exact_authorization_context,
)
from apps.documents.services.hybrid_graph_dependencies import resolve
from apps.documents.services.hybrid_graph_lifecycle import (
    query_embedding as load_query_embedding,
)
from apps.documents.services.hybrid_graph_lifecycle import (
    start_graph,
)
from apps.documents.services.hybrid_graph_orchestration import (
    hybrid_graph_candidate_pool,  # noqa: F401 - retained scored-flow patch point
)
from apps.documents.services.source_loading import (
    current_source_runtime,
    source_mode_enabled,
)
from lib.replay_observation import retrieval_stages
from lib.retrieval_redaction import RetrievalLogReason, retrieval_log_fields

from .chunk_search_lifetime import enter_search, graph_preflight
from .chunk_search_revalidation import ranking_status, reconcile_candidate_order
from .chunk_search_scoped_rows import ranked_rows
from .chunk_search_scored import _scored_search
from .closed_search_finalizer import _bind_search_authorizer
from .native_search_schedule import acquisition_schedule

if TYPE_CHECKING:
    from apps.documents.models.chunks import TextChunk

logger = structlog.stdlib.get_logger(__name__)
_EVALUATION_GRAPH_FAILURE = _legacy_graph.EVALUATION_GRAPH_FAILURE
_EVALUATION_GRAPH_MISS = _legacy_graph.EVALUATION_GRAPH_MISS
_apply_graph_overlay = _legacy_graph.apply_graph_overlay
_graph_diagnostics = _legacy_graph.graph_diagnostics

_exact_term_query = _candidate_exact_term_query
_salient_exact_terms = _candidate_salient_exact_terms


@dataclass(frozen=True, slots=True)
class CandidateRankingResult:
    combined_candidates: tuple[object, ...]
    graph_candidates: tuple[object, ...]
    ranked_results: tuple[object, ...]
    inaccessible_candidate_count: int
    materialization_ms: float
    rerank_ms: float
    score_set: RerankScoreSet | None = None
    rerank_status: str = "unknown"


def materialize_and_rerank_candidates(
    model_cls: type[TextChunk],
    query: str,
    top_k: int,
    baseline_candidates: tuple[object, ...],
    *,
    authorized_scope: object | None,
    graph_chunk_ids: tuple[int, ...] = (),
    max_graph_candidates: int = 0,
    force_complete_rerank: bool = False,
    capture_scores: bool = False,
    _eval_rerank_capability: object | None = None,
) -> CandidateRankingResult:
    """Permission-refetch graph rows, append to the baseline, and rerank once."""

    baseline_identities = validate_candidate_request(
        query,
        top_k,
        baseline_candidates,
        graph_chunk_ids,
        max_graph_candidates,
        force_complete_rerank,
        capture_scores,
        _eval_rerank_capability,
    )

    materialization_started = perf_counter()
    inaccessible = 0
    allowed_doc_ids: tuple[UUID, ...] = ()
    if authorized_scope is not None:
        from apps.documents.services.chunk_search_candidates import (
            AuthorizedDocumentScope,
        )

        if type(authorized_scope) is not AuthorizedDocumentScope:
            raise ValueError("authorized_scope must be an exact frozen scope")
        allowed_doc_ids = authorized_scope.allowed_doc_ids
        inaccessible += sum(
            getattr(row, "doc_id", None) not in set(allowed_doc_ids)
            for row in baseline_candidates
        )
    elif graph_chunk_ids:
        inaccessible += len(graph_chunk_ids)

    baseline_id_set = {
        identifier
        for identity_type, identifier in baseline_identities
        if identity_type == "pk"
    }
    novel_ids = tuple(
        identifier
        for identifier in graph_chunk_ids
        if identifier not in baseline_id_set
    )
    graph_rows: tuple[object, ...] = ()
    if novel_ids and authorized_scope is not None:
        graph_rows, rejected = materialize_graph_rows(
            model_cls, novel_ids, allowed_doc_ids
        )
        inaccessible += rejected

    combined = reconcile_candidate_order((*baseline_candidates, *graph_rows))
    materialization_ms = (perf_counter() - materialization_started) * 1_000
    rerank_started = perf_counter()
    score_set = None
    if _eval_rerank_capability is _STRICT_EVALUATION_RERANK:
        reranked = _strict_local_rerank_chunks(
            model_cls,
            query,
            combined,
            top_k,
            _capability=_eval_rerank_capability,
        )
    elif not force_complete_rerank and len(combined) <= top_k:
        reranked = _fallback_rerank(model_cls, combined, top_k)
    elif capture_scores:
        runtime = current_source_runtime()
        scored = rerank_chunks_scored(
            model_cls,
            query,
            combined,
            top_k,
            **({"turn_budget": runtime.budget} if runtime else {}),
        )
        by_id = {_candidate_identifier(row): row for row in combined}
        if (
            not scored.ranked_ids
            or len(scored.ranked_ids) > top_k
            or len(set(scored.ranked_ids)) != len(scored.ranked_ids)
            or len(by_id) != len(combined)
            or any(type(pk) is not int or pk not in by_id for pk in scored.ranked_ids)
        ):
            reranked = _fallback_rerank(model_cls, combined, top_k)
        else:
            reranked = [by_id[pk] for pk in scored.ranked_ids]
            score_set = scored.score_set
    else:
        reranked = rerank_chunks(model_cls, query, combined, top_k)
    status = ranking_status(reranked, score_set)
    ranked_results = ranked_rows(reranked)
    rerank_ms = (perf_counter() - rerank_started) * 1_000
    combined_identities = set(_candidate_identity(row) for row in combined)
    ranked_identities = tuple(_candidate_identity(row) for row in ranked_results)
    if len(set(ranked_identities)) != len(ranked_identities) or any(
        identity not in combined_identities for identity in ranked_identities
    ):
        raise ValueError("reranker returned rows outside the candidate pool")
    inaccessible += sum(
        getattr(row, "doc_id", None) not in set(allowed_doc_ids)
        for row in ranked_results
        if authorized_scope is not None
    )
    return CandidateRankingResult(
        combined_candidates=tuple(combined),
        graph_candidates=graph_rows,
        ranked_results=ranked_results,
        inaccessible_candidate_count=inaccessible,
        materialization_ms=materialization_ms,
        rerank_ms=rerank_ms,
        score_set=score_set,
        rerank_status=status,
    )


def text_chunk_search(
    model_cls: type[TextChunk],
    query: str,
    top_k: int,
    docs: list,
    *,
    authorization_context: object | None = None,
    hybrid_graph_dependencies: HybridGraphRetrievalDependencies | None = None,
):
    from apps.chat.services.rag_config import (
        evidence_selection_config,
        rag_preservation_config,
    )

    if source_mode_enabled() and current_source_runtime() is None:
        from apps.documents.services.source_loading import SourcePreparationLimited

        raise SourcePreparationLimited("source search requires shared ledger")
    total_start = perf_counter()
    graph_handle = None
    lifetime = None
    try:
        capture_scores = (
            evidence_selection_config().mode == "adaptive"
            or source_mode_enabled()
            or rag_preservation_config().rerank_text_mode == "windowed"
        )
        overlay_enabled = bool(getattr(django_settings, "KG_OVERLAY_ENABLED", False))
        lifetime, retrieval_scope = enter_search(
            model_cls,
            docs,
            authorization_context,
            query,
            overlay_enabled,
        )
        hybrid_graph_dependencies, hybrid_requested = resolve(
            overlay_enabled, authorization_context, hybrid_graph_dependencies
        )

        graph_config, graph_scope, graph_preflight_status, search_documents = (
            graph_preflight(
                overlay_enabled,
                hybrid_requested,
                docs,
                authorization_context,
                retrieval_scope,
                freeze_authorized_document_scope,
                dependencies=hybrid_graph_dependencies,
            )
        )

        graph_handle = start_graph(
            query,
            authorization_context,
            hybrid_graph_dependencies,
            overlay_enabled and hybrid_requested and graph_preflight_status is None,
        )
        initial_vector_error, query_embedding = None, None
        try:
            query_embedding = load_query_embedding(query)
            if getattr(query_embedding, "compatibility_key", None) is None:
                initial_vector_error = RetrievalLogReason.EMBEDDING_UNAVAILABLE.value
        except Exception:
            initial_vector_error = RetrievalLogReason.EMBEDDING_UNAVAILABLE.value
            logger.warning(
                "obs.rag.vector_search_failed",
                **retrieval_log_fields(
                    reason=RetrievalLogReason.EMBEDDING_UNAVAILABLE,
                    count=0,
                    elapsed_ms=0.0,
                ),
            )

        snapshot = collect_hybrid_candidate_snapshot(
            model_cls,
            query,
            top_k,
            search_documents,
            query_embedding=query_embedding,
            graph_config=graph_config if graph_preflight_status is None else None,
            initial_vector_error=initial_vector_error,
            app_config_getter=apps.get_app_config,
            authorization_context=(
                authorization_context
                if hybrid_requested and graph_preflight_status is None
                else None
            ),
            _graph_dependencies=(hybrid_graph_dependencies if overlay_enabled
                and hybrid_requested and graph_preflight_status is None else None),
        )
        requested_schedule = acquisition_schedule()
        native = None
        if requested_schedule == "native_closure" and (
            not overlay_enabled
            or (
                hybrid_requested
                and graph_preflight_status is None
                and type(hybrid_graph_dependencies) is HybridGraphRetrievalDependencies
                and getattr(
                    hybrid_graph_dependencies.settings, "graph_remediation_mode", None
                ) == "v2"
                and is_exact_authorization_context(authorization_context)
            )
        ):
            from .native_graph_search_closure import _try_native_search_closure

            authorized_rows = _bind_search_authorizer(
                authorized_search_rows, authorization_context, hybrid_requested
            )
            native = _try_native_search_closure(
                snapshot, query, top_k,
                authorize=authorized_rows,
                dependencies=hybrid_graph_dependencies,
                overlay_enabled=overlay_enabled,
            )
        if native is not None:
            snapshot, ranking = native.snapshot, native.ranking
            final_rows, score_set = native.final_rows, native.score_set
            hybrid_pool, graph_diagnostics = (
                native.hybrid_pool, native.graph_diagnostics
            )
        else:
            snapshot, ranking, final_rows, score_set, hybrid_pool, graph_diagnostics = (
                _scored_search(
                    snapshot, query, top_k, model_cls=model_cls,
                    authorization_context=authorization_context,
                    hybrid_graph_dependencies=hybrid_graph_dependencies,
                    hybrid_requested=hybrid_requested, overlay_enabled=overlay_enabled,
                    graph_config=graph_config, graph_scope=graph_scope,
                    graph_preflight_status=graph_preflight_status,
                    graph_handle=graph_handle, total_start=total_start,
                    capture_scores=capture_scores,
                )
            )
        reranked_results = final_rows[2]

        total_ms = min(300_000.0, max(0.0, (perf_counter() - total_start) * 1000))
        diagnostics = completed_search_diagnostics(
            logger,
            model_cls,
            snapshot,
            reranked_results,
            score_set,
            graph_diagnostics,
            overlay_enabled=overlay_enabled,
            docs_count=len(docs),
            elapsed_ms=total_ms,
        )
        retrieval_stages(
            query,
            top_k,
            snapshot,
            ranking,
            reranked_results,
            graph_diagnostics,
            overlay_enabled=overlay_enabled,
            hybrid_pool=hybrid_pool,
            graph_seed_attempted=(
                graph_config is not None and graph_preflight_status is None
            ),
            acquisition_schedule_requested=requested_schedule,
            acquisition_schedule_observed=(
                native.acquisition_schedule if native else "scored"
            ),
            acquisition_initial_ineligibility_reason=(
                "native_ineligible"
                if requested_schedule == "native_closure" and native is None else None
            ),
            acquisition_score_status=(
                "not_requested" if native else
                "complete" if ranking.rerank_status == "complete" else "unavailable"
            ),
            materialized_union_status=(
                native.materialized_union_status if native else "observed"
            ),
            preclosure_pool_count=(native.preclosure_pool_count if native else None),
        )
        return (*final_rows, diagnostics)
    except Exception as error:
        log_search_failure(logger, error)
        raise
    finally:
        if graph_handle is not None:
            graph_handle.close()
        if lifetime is not None:
            lifetime.close()


__all__ = ["materialize_and_rerank_candidates", "text_chunk_search"]
