"""Best-effort private replay events; never a retrieval or logging path."""

from lib.evidence_observation import active, mark_failed, publish


def _emit(event, make_data):
    if not active():
        return
    try:
        publish(event, make_data())
    except BaseException:
        # Even a malformed observation must not alter the application turn.
        mark_failed()


def prepared_queries(queries, requested_top_k, candidate_top_k):
    _emit(
        "rag_prepared_queries",
        lambda: {
            "queries": list(queries),
            "requested_top_k": requested_top_k,
            "candidate_top_k": candidate_top_k,
        },
    )


def search_outcomes(queries, outcomes):
    def data():
        return {
            "queries": [
                {
                    "query": query,
                    "status": (
                        "ok"
                        if isinstance(outcome, dict)
                        else "timeout"
                        if isinstance(outcome, TimeoutError)
                        else "error"
                    ),
                }
                for query, outcome in zip(queries, outcomes, strict=True)
            ]
        }

    _emit("rag_search_outcomes", data)


def final_selection(packet, max_passages, token_budget):
    _emit(
        "rag_final_selection",
        lambda: {
            "rows": [dict(row) for row in packet.chunks],
            "limits": {"max_passages": max_passages, "token_budget": token_budget()},
        },
    )


def retrieval_stages(
    query,
    top_k,
    snapshot,
    ranking,
    reranked,
    diagnostics,
    *,
    overlay_enabled,
    hybrid_pool,
    graph_seed_attempted,
):
    def row_data(row, rank):
        doc_id = str(row.doc_id)
        chunk_id = row.pk
        return {
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "rank": rank,
            "text": row.content,
            "citation": f"[doc:{doc_id} chunk:{chunk_id}]",
        }

    def rows_data(rows):
        return [row_data(row, rank) for rank, row in enumerate(rows, 1)]

    def make_data():
        reasons = [
            diagnostics[key]
            for key in ("graph_direct_reason", "graph_extended_reason")
            if key in diagnostics and diagnostics[key] not in (None, "none", "not_run")
        ]
        readiness_failures = reasons.count("readiness_mismatch") or None
        graph_rows = None
        if overlay_enabled:
            if hybrid_pool is not None:
                baseline_ids = {row.pk for row in snapshot.baseline_candidates}
                graph_rows = [row for row in hybrid_pool if row.pk not in baseline_ids]
            else:
                graph_rows = ranking.graph_candidates
        branch_statuses = {
            branch: diagnostics[f"graph_{branch}_status"]
            for branch in ("direct", "extended")
            if diagnostics.get(f"graph_{branch}_status")
            in (
                "not_run",
                "failed",
                "succeeded_new",
                "succeeded_duplicates",
                "succeeded_empty",
                "succeeded_unmaterialized",
            )
        }
        graph_status = diagnostics.get("graph_status")
        limits = getattr(snapshot, "effective_limits", None)
        return {
            "query": query,
            "requested_limits": {"top_k": top_k},
            "effective_limits": (
                {
                    "vector": limits.vector,
                    "trigram": limits.trigram,
                    "exact": limits.exact,
                    "trigram_similarity_min": limits.trigram_similarity_min,
                }
                if limits is not None
                else None
            ),
            "baseline_branches": {
                "vector": rows_data(snapshot.vector_results),
                "trigram": rows_data(snapshot.trigram_results),
                "exact": rows_data(snapshot.exact_results),
            },
            "graph": {
                "ready": None,
                "status": (
                    graph_status if graph_status in ("hit", "miss", "error") else None
                ),
                "reasons": reasons if reasons else None,
                "branch_statuses": branch_statuses or None,
                "candidate_provenance": (
                    "novel_fused"
                    if hybrid_pool is not None
                    else "materialized_novel"
                    if graph_rows is not None
                    else None
                ),
                "seeds": (
                    [
                        {"chunk_id": seed.chunk_id, "rank": seed.rank}
                        for seed in snapshot.graph_seeds
                    ]
                    if graph_seed_attempted and not snapshot.graph_seed_error
                    else None
                ),
                "candidates": rows_data(graph_rows) if graph_rows is not None else None,
            },
            "materialized_union": rows_data(ranking.combined_candidates),
            "post_rerank": rows_data(reranked),
            "readiness_failure_count": readiness_failures,
        }

    _emit("retrieval_stages", make_data)
