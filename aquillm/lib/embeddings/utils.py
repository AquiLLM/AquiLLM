"""
Embedding utility functions.
"""

import logging

from .config import get_target_dims

logger = logging.getLogger(__name__)


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
            "Embedding dims (%d) exceed APP_EMBED_DIMS (%d); truncating.",
            current,
            target_dims,
        )
        return embedding[:target_dims]
    logger.warning(
        "Embedding dims (%d) below APP_EMBED_DIMS (%d); zero-padding.",
        current,
        target_dims,
    )
    return embedding + [0.0] * (target_dims - current)


__all__ = ['fit_embedding_dims']
