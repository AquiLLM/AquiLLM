"""Handle partial results from the existing parallel direct-RAG searches."""


async def run_parallel_searches(consumer, queries, candidate_top_k, *, vector_runner):
    """Keep each worker's DB cleanup and cancellation on the original turn."""
    import asyncio
    from django.db import connections
    from lib.llm.turn_context import bounded_retrieval, current_turn

    def search_worker(search_query):
        try:
            return vector_runner(consumer, search_query, candidate_top_k)
        finally:
            connections.close_all()

    return await bounded_retrieval(
        asyncio.gather(
            *(asyncio.to_thread(search_worker, query) for query in queries),
            return_exceptions=True,
        ),
        current_turn().budget,
    )


def successful_search_results(outcomes, logger):
    results = [outcome for outcome in outcomes if isinstance(outcome, dict)]
    failed_count = len(outcomes) - len(results)
    if not results:
        first_error = next(
            (outcome for outcome in outcomes if isinstance(outcome, BaseException)),
            RuntimeError("all direct-RAG retrieval queries failed"),
        )
        raise first_error
    if failed_count:
        logger.warning(
            "obs.rag.partial_retrieval_failure",
            failed_count=failed_count,
            total_count=len(outcomes),
        )
    return results
