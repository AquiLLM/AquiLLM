"""Handle partial results from the existing parallel direct-RAG searches."""


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
