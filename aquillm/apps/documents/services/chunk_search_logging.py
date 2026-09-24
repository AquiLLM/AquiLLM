"""Literal redacted events for acquisition failures."""

from django.core.exceptions import ValidationError
from django.db import DatabaseError

from lib.retrieval_redaction import RetrievalLogReason, retrieval_log_fields


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
