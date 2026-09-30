"""Literal redacted events and diagnostics for chunk search."""

from django.core.exceptions import ValidationError
from django.db import DatabaseError

from apps.documents.services.chunk_rerank_score_transport import serialize_score_set
from lib.retrieval_redaction import RetrievalLogReason, retrieval_log_fields


def completed_search_diagnostics(
    logger,
    model_cls,
    snapshot,
    reranked_results,
    score_set,
    graph_diagnostics,
    *,
    overlay_enabled,
    docs_count,
    elapsed_ms,
):
    logger.info(
        "obs.rag.search",
        **retrieval_log_fields(
            reason=RetrievalLogReason.COMPLETED,
            count=0,
            elapsed_ms=elapsed_ms,
        ),
    )
    chunks_with_embeddings = None
    if not reranked_results:
        try:
            chunks_with_embeddings = (
                model_cls.objects.filter_by_documents(snapshot.documents)
                .exclude(embedding__isnull=True)
                .count()
            )
        except Exception:
            logger.warning(
                "obs.rag.chunk_count_failed",
                **retrieval_log_fields(
                    reason=RetrievalLogReason.INTERNAL_FAILURE,
                    count=0,
                    elapsed_ms=0.0,
                ),
            )
    diagnostics = {
        "doc_count": docs_count,
        "chunks_with_embeddings": chunks_with_embeddings,
        "vector_error": snapshot.vector_error,
        "trigram_candidates": len(snapshot.trigram_chunk_ids),
        "exact_term_count": len(snapshot.exact_terms),
    }
    if overlay_enabled:
        diagnostics.update(graph_diagnostics)
    if score_set is not None and reranked_results:
        diagnostics["_score_set"] = serialize_score_set(score_set)
    if not reranked_results:
        logger.info(
            "obs.rag.search_empty",
            **retrieval_log_fields(
                reason=RetrievalLogReason.NO_SEEDS,
                count=0,
                elapsed_ms=elapsed_ms,
            ),
        )
    return diagnostics


def log_search_failure(logger, error):
    if isinstance(error, DatabaseError):
        logger.error(
            "obs.rag.search_db_error",
            **retrieval_log_fields(
                reason=RetrievalLogReason.UPSTREAM_UNAVAILABLE, count=0, elapsed_ms=0.0
            ),
        )
    elif isinstance(error, ValidationError):
        logger.error(
            "obs.rag.search_validation_error",
            **retrieval_log_fields(
                reason=RetrievalLogReason.INVALID_REQUEST, count=0, elapsed_ms=0.0
            ),
        )
    else:
        logger.error(
            "obs.rag.search_error",
            **retrieval_log_fields(
                reason=RetrievalLogReason.INTERNAL_FAILURE, count=0, elapsed_ms=0.0
            ),
        )
