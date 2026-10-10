"""
Embedding utility functions.
"""

from math import hypot, isfinite

import structlog

from .config import get_target_dims

logger = structlog.stdlib.get_logger(__name__)


class EmbeddingContractError(ValueError):
    """Permanent malformed provider result; never retry or cross vector spaces."""


def validate_embedding(embedding: list[float]) -> None:
    """Require a finite numeric vector with a usable nonzero magnitude."""
    if not isinstance(embedding, (list, tuple)) or not embedding:
        raise EmbeddingContractError("Embedding must be a nonempty numeric vector")
    try:
        valid = all(
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and isfinite(value)
            for value in embedding
        )
        norm = hypot(*embedding) if valid else 0.0
    except (OverflowError, TypeError):
        valid, norm = False, 0.0
    if not valid or not isfinite(norm) or norm == 0.0:
        raise EmbeddingContractError("Embedding must be finite, numeric, and nonzero")


def fit_embedding_dims(embedding: list[float]) -> list[float]:
    """
    Fit embedding vectors to the pgvector schema dimension.
    Existing DB columns are vector(1024), so pad/truncate as needed.
    """
    validate_embedding(embedding)
    embedding = list(embedding)
    target_dims = get_target_dims()

    current = len(embedding)
    if current == target_dims:
        return embedding
    if current > target_dims:
        logger.warning(
            "obs.embed.dims_truncate",
            current_dims=current,
            target_dims=target_dims,
        )
        fitted = embedding[:target_dims]
        validate_embedding(fitted)
        return fitted
    logger.warning(
        "obs.embed.dims_pad",
        current_dims=current,
        target_dims=target_dims,
    )
    return embedding + [0.0] * (target_dims - current)


__all__ = ["EmbeddingContractError", "fit_embedding_dims", "validate_embedding"]
