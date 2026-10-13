"""Bounded Cohere score collection for the public rerank dispatcher."""


def score_cohere(cohere, query, chunks_list, top_k, *, state, turn_budget,
                 candidate_ids, query_fp, pool, unavailable):
    # Resolve the dispatcher hooks at call time, including monkeypatches.
    from . import chunk_rerank as api

    try:
        from math import isfinite

        from .chunk_rerank_config import rerank_timeout_seconds

        width = min(6, turn_budget.limits.in_flight_pairs)
        if width <= 0:
            return unavailable(candidate_ids[:top_k])
        all_results = []
        for start in range(0, len(chunks_list), width):
            batch = chunks_list[start : start + width]

            def send(remaining):
                response = cohere.rerank(
                    model="rerank-english-v3.0",
                    query=query,
                    documents=[
                        {"content": chunk.content, "id": chunk.pk} for chunk in batch
                    ],
                    rank_fields=["content"],
                    top_n=len(batch),
                    return_documents=True,
                    request_options={
                        "timeout_in_seconds": min(rerank_timeout_seconds(), remaining),
                        "max_retries": 0,
                    },
                )
                results = tuple(response.results)
                if len(results) != len(batch) or {r.document.id for r in results} != {
                    c.pk for c in batch
                }:
                    return None
                if any(
                    type(getattr(r, "relevance_score", None)) not in (int, float)
                    or not isfinite(float(r.relevance_score))
                    for r in results
                ):
                    return None
                return results

            results = state.run(len(batch), send, timeout=rerank_timeout_seconds())
            if results is None:
                return unavailable(candidate_ids[:top_k])
            all_results.extend(results)
        if not state.can_admit():
            return unavailable(candidate_ids[:top_k])
        ranked_ids = tuple(result.document.id for result in all_results)
        by_id = {chunk.pk: chunk for chunk in chunks_list}
        if (
            len(ranked_ids) != len(candidate_ids)
            or len(set(ranked_ids)) != len(candidate_ids)
            or set(ranked_ids) != set(candidate_ids)
        ):
            return unavailable(candidate_ids[:top_k])
        scores_by_id = {}
        for result in all_results:
            value = getattr(result, "relevance_score", None)
            if type(value) not in (int, float):
                return unavailable(candidate_ids[:top_k])
            chunk = by_id[result.document.id]
            scores_by_id[chunk.pk] = api.PassageScore(
                chunk.pk,
                chunk.doc_id,
                chunk.chunk_number,
                api.fingerprint_text(chunk.content),
                api.fingerprint_pair(query, chunk.content),
                float(value),
            )

        if not all(isfinite(score.value) for score in scores_by_id.values()):
            return unavailable(candidate_ids[:top_k])
        scores = tuple(scores_by_id[pk] for pk in candidate_ids)
        stable_ranked = tuple(
            sorted(
                candidate_ids,
                key=lambda pk: (-scores_by_id[pk].value, candidate_ids.index(pk)),
            )
        )
        return api.ScoredRerankResult(
            stable_ranked[:top_k],
            api.RerankScoreSet(
                "v2",
                query_fp,
                api.fingerprint_text("cohere:rerank-english-v3.0"),
                pool,
                "listwise",
                "complete",
                candidate_ids,
                scores,
            ),
        )
    except Exception:
        api.logger.warning(
            "obs.rag.cohere_rerank_failed",
            **api.retrieval_log_fields(
                reason=api.RetrievalLogReason.UPSTREAM_UNAVAILABLE, count=0, elapsed_ms=0.0
            ),
        )
        return unavailable(candidate_ids[:top_k])
