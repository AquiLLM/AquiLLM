"""Fail-open cache I/O with preservation-specific finite transports."""

from typing import Any

import structlog

from .bounded_rag_cache import cache_operation
from .source_loading import current_source_runtime

logger = structlog.stdlib.get_logger(__name__)


def cache_get(cache, key: str) -> Any | None:
    if current_source_runtime() is not None:
        return cache_operation("get", key)
    try:
        return cache.get(key)
    except Exception as exc:
        logger.warning(
            "obs.rag.cache_get_failed",
            error_type=type(exc).__name__,
        )
        return None


def cache_set(cache, key: str, value: Any, timeout: int) -> None:
    if current_source_runtime() is not None:
        cache_operation("set", key, value, timeout=timeout)
        return
    try:
        cache.set(key, value, timeout=timeout)
    except Exception as exc:
        logger.warning(
            "obs.rag.cache_set_failed",
            error_type=type(exc).__name__,
        )
