"""Mem0 client management for cloud and OSS SDK."""

from __future__ import annotations

import asyncio
from os import getenv

import structlog

from .graph_payloads import normalize_graph_search_results_payload
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
_MEM0_OSS_VECTOR = None
_MEM0_OSS_VECTOR_INIT_ATTEMPTED = False

_MEM0_CLIENT_ASYNC = None
_MEM0_CLIENT_ASYNC_INIT_ATTEMPTED = False
_MEM0_OSS_ASYNC = None
_MEM0_OSS_ASYNC_INIT_ATTEMPTED = False
_MEM0_OSS_ASYNC_VECTOR = None
_MEM0_OSS_ASYNC_VECTOR_INIT_ATTEMPTED = False

_mem0_client_async_lock: asyncio.Lock | None = None
_mem0_oss_async_lock: asyncio.Lock | None = None
_mem0_oss_async_vector_lock: asyncio.Lock | None = None


def _register_memgraph_compat_provider() -> None:
    """Route Mem0's memgraph provider through our local compatibility shim."""
    try:
        from mem0.utils.factory import GraphStoreFactory  # type: ignore

        GraphStoreFactory.provider_to_class["memgraph"] = (
            "lib.memory.mem0.memgraph_compat.CompatibleMemgraphMemoryGraph"
        )
    except Exception:
        # If Mem0 is unavailable we will fall through to the normal init error path.
        pass

    try:
        from mem0.memory.memgraph_memory import MemoryGraph  # type: ignore

        if getattr(MemoryGraph, "_aquillm_payload_normalizer_patched", False):
            return

        original = MemoryGraph._retrieve_nodes_from_data

        def _wrapped_retrieve_nodes_from_data(self, search_results, *args, **kwargs):
            normalized = normalize_graph_search_results_payload(search_results)
            return original(self, normalized, *args, **kwargs)

        MemoryGraph._retrieve_nodes_from_data = _wrapped_retrieve_nodes_from_data
        MemoryGraph._aquillm_payload_normalizer_patched = True
    except Exception:
        # If Mem0 is unavailable we will fall through to the normal init error path.
        pass


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


def _get_mem0_oss_async_vector_lock() -> asyncio.Lock:
    global _mem0_oss_async_vector_lock
    if _mem0_oss_async_vector_lock is None:
        _mem0_oss_async_vector_lock = asyncio.Lock()
    return _mem0_oss_async_vector_lock


def _build_mem0_oss_config_dict(graph_enabled_override: bool | None = None):
    return build_mem0_oss_config_dict(graph_enabled_override=graph_enabled_override)


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
        logger.warning("obs.memory.mem0_no_key")
        return None

    try:
        from mem0 import MemoryClient  # type: ignore

        _MEM0_CLIENT = MemoryClient(api_key=api_key)
        return _MEM0_CLIENT
    except Exception as exc:
        logger.warning("obs.memory.mem0_init_error", error_type=type(exc).__name__, error=str(exc))
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
            logger.warning("obs.memory.mem0_async_init_error", error_type=type(exc).__name__, error=str(exc))
            return None


def _handle_oss_init_exception(exc: Exception):
    if _env_bool("MEM0_GRAPH_ENABLED", False) and (not _env_bool("MEM0_GRAPH_FAIL_OPEN", True)):
        raise exc
    logger.warning("obs.memory.mem0_oss_init_error", error_type=type(exc).__name__, error=str(exc))
    return None


def get_mem0_oss():
    """Create a local OSS Mem0 SDK client once. Returns None when unavailable/misconfigured."""
    global _MEM0_OSS, _MEM0_OSS_INIT_ATTEMPTED
    if _MEM0_OSS_INIT_ATTEMPTED:
        return _MEM0_OSS
    _MEM0_OSS_INIT_ATTEMPTED = True

    try:
        from mem0 import Memory  # type: ignore

        _register_memgraph_compat_provider()
        config, clear_dims = _build_mem0_oss_config_dict()
        _MEM0_OSS = Memory.from_config(config)  # type: ignore[attr-defined]
        if clear_dims:
            _clear_mem0_embedding_dims_override(_MEM0_OSS)
        return _MEM0_OSS
    except Exception as exc:
        return _handle_oss_init_exception(exc)


def get_mem0_oss_vector():
    """Create a local OSS Mem0 SDK client configured without graph_store."""
    global _MEM0_OSS_VECTOR, _MEM0_OSS_VECTOR_INIT_ATTEMPTED
    if _MEM0_OSS_VECTOR_INIT_ATTEMPTED:
        return _MEM0_OSS_VECTOR
    _MEM0_OSS_VECTOR_INIT_ATTEMPTED = True

    try:
        from mem0 import Memory  # type: ignore

        _register_memgraph_compat_provider()
        config, clear_dims = _build_mem0_oss_config_dict(graph_enabled_override=False)
        _MEM0_OSS_VECTOR = Memory.from_config(config)  # type: ignore[attr-defined]
        if clear_dims:
            _clear_mem0_embedding_dims_override(_MEM0_OSS_VECTOR)
        return _MEM0_OSS_VECTOR
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

            _register_memgraph_compat_provider()
            config, clear_dims = _build_mem0_oss_config_dict()
            _MEM0_OSS_ASYNC = await AsyncMemory.from_config(config)  # type: ignore[attr-defined]
            if clear_dims:
                _clear_mem0_embedding_dims_override(_MEM0_OSS_ASYNC)
            return _MEM0_OSS_ASYNC
        except Exception as exc:
            return _handle_oss_init_exception(exc)


async def get_mem0_oss_async_vector():
    """Async local OSS Mem0 SDK client configured without graph_store."""
    global _MEM0_OSS_ASYNC_VECTOR, _MEM0_OSS_ASYNC_VECTOR_INIT_ATTEMPTED
    async with _get_mem0_oss_async_vector_lock():
        if _MEM0_OSS_ASYNC_VECTOR_INIT_ATTEMPTED:
            return _MEM0_OSS_ASYNC_VECTOR
        _MEM0_OSS_ASYNC_VECTOR_INIT_ATTEMPTED = True

        try:
            from mem0 import AsyncMemory  # type: ignore

            _register_memgraph_compat_provider()
            config, clear_dims = _build_mem0_oss_config_dict(graph_enabled_override=False)
            _MEM0_OSS_ASYNC_VECTOR = await AsyncMemory.from_config(config)  # type: ignore[attr-defined]
            if clear_dims:
                _clear_mem0_embedding_dims_override(_MEM0_OSS_ASYNC_VECTOR)
            return _MEM0_OSS_ASYNC_VECTOR
        except Exception as exc:
            return _handle_oss_init_exception(exc)


__all__ = [
    "get_mem0_client",
    "get_mem0_client_async",
    "get_mem0_oss",
    "get_mem0_oss_vector",
    "get_mem0_oss_async",
    "get_mem0_oss_async_vector",
    "_build_mem0_oss_config_dict",
    "_normalize_openai_base_url",
    "_clear_mem0_embedding_dims_override",
]
