"""Existing scored orchestration, retaining chunk_search compatibility patch points."""


def _scored_search(snapshot, query, top_k, *, model_cls, authorization_context,
                   hybrid_graph_dependencies, hybrid_requested, overlay_enabled,
                   graph_config, graph_scope, graph_preflight_status, graph_handle,
                   total_start, capture_scores):
    from . import chunk_search as search

    graph_chunk_ids: tuple[int, ...] = ()
    graph_diagnostics: dict[str, object] = {}
    hybrid_pool: tuple[object, ...] | None = None
    if overlay_enabled and hybrid_requested:
        if (
            graph_preflight_status is None
            and type(hybrid_graph_dependencies) is (
                search.HybridGraphRetrievalDependencies
            )
            and search.is_exact_authorization_context(authorization_context)
        ):
            hybrid_pool, graph_diagnostics = search.hybrid_graph_candidate_pool(
                snapshot,
                query,
                authorization_context,
                hybrid_graph_dependencies,
                handle=graph_handle,
            )
        else:
            hybrid_pool = snapshot.baseline_candidates
            graph_diagnostics = search._graph_diagnostics(
                started_at=total_start,
                seed_count=len(snapshot.graph_seeds),
                candidate_count=0,
                status=graph_preflight_status or "error",
                algorithm_signature=getattr(
                    graph_config, "algorithm_signature", None
                ),
                version_signature=None,
            )
    elif overlay_enabled:
        graph_chunk_ids, graph_diagnostics = search._apply_graph_overlay(
            model_cls,
            snapshot,
            graph_scope,
            graph_config,
            preflight_status=graph_preflight_status,
        )

    if (getattr(snapshot, "source_admission", None) is not None
        and hybrid_pool is not None):
        snapshot = snapshot.source_admission.refresh_snapshot(snapshot, hybrid_pool)

    authorized_rows = search._bind_search_authorizer(
        search.authorized_search_rows, authorization_context, hybrid_requested
    )
    if hybrid_pool is not None:
        ranking = search.materialize_and_rerank_candidates(
            model_cls,
            query,
            top_k,
            authorized_rows(hybrid_pool),
            authorized_scope=None,
            force_complete_rerank=graph_diagnostics.get("graph_status") == "hit",
            capture_scores=capture_scores,
        )
    else:
        try:
            ranking = search.materialize_and_rerank_candidates(
                model_cls,
                query,
                top_k,
                authorized_rows(snapshot.baseline_candidates),
                authorized_scope=graph_scope,
                graph_chunk_ids=graph_chunk_ids,
                max_graph_candidates=(
                    int(getattr(graph_config, "max_candidates", 0))
                    if graph_chunk_ids
                    else 0
                ),
                capture_scores=capture_scores,
            )
        except Exception:
            ranking = search.materialize_and_rerank_candidates(
                model_cls,
                query,
                top_k,
                authorized_rows(snapshot.baseline_candidates),
                authorized_scope=None,
                capture_scores=capture_scores,
            )
            if overlay_enabled:
                graph_diagnostics = search._graph_diagnostics(
                    started_at=total_start,
                    seed_count=len(snapshot.graph_seeds),
                    candidate_count=0,
                    status="error",
                    algorithm_signature=getattr(
                        graph_config,
                        "algorithm_signature",
                        None,
                    ),
                    version_signature=None,
                )
    if overlay_enabled and ranking.inaccessible_candidate_count:
        graph_diagnostics.update(
            graph_status="error",
            graph_candidate_count=0,
        )
    if overlay_enabled:
        graph_diagnostics["graph_ms"] = (
            float(graph_diagnostics.get("graph_ms", 0.0))
            + ranking.materialization_ms
        )
        if not hybrid_requested and graph_diagnostics.get("graph_status") == "hit":
            graph_diagnostics["graph_candidate_count"] = len(
                ranking.graph_candidates
            )
    snapshot, final_rows, score_set = search.finalize_ranking(
        snapshot,
        ranking,
        authorize=authorized_rows,
        query=query,
        top_k=top_k,
    )
    return snapshot, ranking, final_rows, score_set, hybrid_pool, graph_diagnostics
