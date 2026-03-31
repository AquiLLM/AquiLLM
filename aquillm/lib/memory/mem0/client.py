"""Mem0 client management for cloud and OSS SDK."""

from __future__ import annotations

import asyncio
from os import getenv

import structlog

from .config_builder import (
    build_mem0_oss_config_dict,
    clear_mem0_embedding_dims_override,
    normalize_openai_base_url,
)

logger = structlog.stdlib.get_logger(__name__)

_MEM0_CLIENT = None
_MEM0_INIT_ATTEMPTED = False
_MEM0_OSS = None
_MEM0_OSS_INIT_ATTEMPTED = False

_MEM0_CLIENT_ASYNC = None
_MEM0_CLIENT_ASYNC_INIT_ATTEMPTED = False
_MEM0_OSS_ASYNC = None
_MEM0_OSS_ASYNC_INIT_ATTEMPTED = False

_mem0_client_async_lock: asyncio.Lock | None = None
_mem0_oss_async_lock: asyncio.Lock | None = None


def _get_mem0_client_async_lock() -> asyncio.Lock:
    global _mem0_client_async_lock
    if _mem0_client_async_lock is None:
        _mem0_client_async_lock = asyncio.Lock()
    return _mem0_client_async_lock


def _get_mem0_oss_async_lock() -> asyncio.Lock:
    global _mem0_oss_async_lock
    if _mem0_oss_async_lock is None:
        _mem0_oss_async_lock = asyncio.Lock()
    return _mem0_oss_async_lock


def _build_mem0_oss_config_dict():
    return build_mem0_oss_config_dict()


def _normalize_openai_base_url(url: str) -> str:
    return normalize_openai_base_url(url)


def _clear_mem0_embedding_dims_override(value, seen=None) -> None:
    clear_mem0_embedding_dims_override(value, seen=seen)


def _env_bool(name: str, default: bool) -> bool:
    raw = getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def get_mem0_client():
    """Create a Mem0 cloud client once. Returns None when mem0 is unavailable/misconfigured."""
    global _MEM0_CLIENT, _MEM0_INIT_ATTEMPTED
    if _MEM0_INIT_ATTEMPTED:
        return _MEM0_CLIENT
    _MEM0_INIT_ATTEMPTED = True

    api_key = getenv("MEM0_API_KEY")
    if not api_key:
        logger.warning("MEMORY_BACKEND=mem0 but MEM0_API_KEY is not set; falling back to local memory.")
        return None

    try:
        from mem0 import MemoryClient  # type: ignore

        _MEM0_CLIENT = MemoryClient(api_key=api_key)
        return _MEM0_CLIENT
    except Exception as exc:
        logger.warning("Failed to initialize Mem0 client; using local memory. Error: %s", exc)
        return None


async def get_mem0_client_async():
    """Async Mem0 cloud client singleton. Returns None when unavailable or no API key."""
    global _MEM0_CLIENT_ASYNC, _MEM0_CLIENT_ASYNC_INIT_ATTEMPTED
    async with _get_mem0_client_async_lock():
        if _MEM0_CLIENT_ASYNC_INIT_ATTEMPTED:
            return _MEM0_CLIENT_ASYNC
        _MEM0_CLIENT_ASYNC_INIT_ATTEMPTED = True

        api_key = getenv("MEM0_API_KEY")
        if not api_key:
            return None

        try:
            from mem0 import AsyncMemoryClient  # type: ignore

            _MEM0_CLIENT_ASYNC = AsyncMemoryClient(api_key=api_key)
            return _MEM0_CLIENT_ASYNC
        except Exception as exc:
            logger.warning("Failed to initialize async Mem0 cloud client; using local memory. Error: %s", exc)
            return None


def _handle_oss_init_exception(exc: Exception):
    if _env_bool("MEM0_GRAPH_ENABLED", False) and (not _env_bool("MEM0_GRAPH_FAIL_OPEN", True)):
        raise exc
    logger.warning("Failed to initialize OSS Mem0 client; using local memory. Error: %s", exc)
    return None


def get_mem0_oss():
    """Create a local OSS Mem0 SDK client once. Returns None when unavailable/misconfigured."""
    global _MEM0_OSS, _MEM0_OSS_INIT_ATTEMPTED
    if _MEM0_OSS_INIT_ATTEMPTED:
        return _MEM0_OSS
    _MEM0_OSS_INIT_ATTEMPTED = True

    try:
        from mem0 import Memory  # type: ignore

        config, clear_dims = _build_mem0_oss_config_dict()
        _MEM0_OSS = Memory.from_config(config)  # type: ignore[attr-defined]
        if clear_dims:
            _clear_mem0_embedding_dims_override(_MEM0_OSS)
        return _MEM0_OSS
    except Exception as exc:
        return _handle_oss_init_exception(exc)


async def get_mem0_oss_async():
    """Async local OSS Mem0 SDK client singleton."""
    global _MEM0_OSS_ASYNC, _MEM0_OSS_ASYNC_INIT_ATTEMPTED
    async with _get_mem0_oss_async_lock():
        if _MEM0_OSS_ASYNC_INIT_ATTEMPTED:
            return _MEM0_OSS_ASYNC
        _MEM0_OSS_ASYNC_INIT_ATTEMPTED = True

        try:
            from mem0 import AsyncMemory  # type: ignore

            config, clear_dims = _build_mem0_oss_config_dict()
            _MEM0_OSS_ASYNC = await AsyncMemory.from_config(config)  # type: ignore[attr-defined]
            if clear_dims:
                _clear_mem0_embedding_dims_override(_MEM0_OSS_ASYNC)
            return _MEM0_OSS_ASYNC
        except Exception as exc:
            return _handle_oss_init_exception(exc)


__all__ = [
    "get_mem0_client",
    "get_mem0_client_async",
    "get_mem0_oss",
    "get_mem0_oss_async",
    "_build_mem0_oss_config_dict",
    "_normalize_openai_base_url",
    "_clear_mem0_embedding_dims_override",
]
