"""
Embedding utility functions.
"""

import structlog

from .config import get_target_dims

logger = structlog.stdlib.get_logger(__name__)


def fit_embedding_dims(embedding: list[float]) -> list[float]:
    """
    Fit embedding vectors to the pgvector schema dimension.
    Existing DB columns are vector(1024), so pad/truncate as needed.
    """
    target_dims = get_target_dims()

    current = len(embedding)
    if current == target_dims:
        return embedding
    if current > target_dims:
        logger.warning(
            "obs.embed.config_warning",
            action="truncating",
            current_dims=current,
            target_dims=target_dims,
        )
        return embedding[:target_dims]
    logger.warning(
        "obs.embed.config_warning",
        action="zero_padding",
        current_dims=current,
        target_dims=target_dims,
    )
    return embedding + [0.0] * (target_dims - current)


__all__ = ['fit_embedding_dims']
