"""
Mem0 client management for cloud and OSS SDK.
"""

from __future__ import annotations

import asyncio
import structlog
from os import getenv
from typing import Any, Optional

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


def _normalize_openai_base_url(url: str) -> str:
    """Normalize OpenAI-compatible base URL to include /v1 suffix."""
    base = (url or "").strip().rstrip("/")
    if not base:
        return "http://vllm:8000/v1"
    if base.endswith("/v1"):
        return base
    return f"{base}/v1"


def _clear_mem0_embedding_dims_override(value: Any, seen: Optional[set[int]] = None) -> None:
    """
    Clear Mem0 embedder-level embedding_dims overrides in-place.
    Some Mem0 OpenAI embedder variants default this to 1536, which causes
    OpenAI-compatible local models to receive an unsupported `dimensions` param.
    """
    if seen is None:
        seen = set()

    obj_id = id(value)
    if obj_id in seen:
        return
    seen.add(obj_id)

    if isinstance(value, dict):
        if "embedding_dims" in value:
            value["embedding_dims"] = None
        for child in value.values():
            _clear_mem0_embedding_dims_override(child, seen)
        return

    if isinstance(value, (list, tuple, set)):
        for child in value:
            _clear_mem0_embedding_dims_override(child, seen)
        return

    for attr in ("embedding_dims",):
        if hasattr(value, attr):
            try:
                setattr(value, attr, None)
            except Exception:
                pass

    child_dict = getattr(value, "__dict__", None)
    if isinstance(child_dict, dict):
        _clear_mem0_embedding_dims_override(child_dict, seen)


def _build_mem0_oss_config_dict() -> tuple[dict[str, Any], bool]:
    """
    Build Mem0 OSS SDK config dict for Memory / AsyncMemory.

    Returns:
        (config_dict, clear_openai_embed_dims) — when True, run _clear_mem0_embedding_dims_override on the client.
    """
    llm_provider = getenv("MEM0_LLM_PROVIDER", "openai")
    llm_model = getenv("MEM0_LLM_MODEL", "qwen3.5:4b-q8_0")
    llm_base_url = _normalize_openai_base_url(
        getenv(
            "MEM0_LLM_BASE_URL",
            getenv("MEM0_VLLM_BASE_URL", getenv("MEM0_OLLAMA_BASE_URL", "http://vllm:8000/v1")),
        )
    )
    llm_api_key = getenv("MEM0_LLM_API_KEY", getenv("VLLM_API_KEY", "EMPTY"))

    embed_provider = getenv("MEM0_EMBED_PROVIDER", "openai")
    embed_model = getenv("MEM0_EMBED_MODEL", "Qwen/Qwen3-Embedding-4B")
    embed_base_url = _normalize_openai_base_url(
        getenv(
            "MEM0_EMBED_BASE_URL",
            getenv("MEM0_VLLM_BASE_URL", getenv("MEM0_OLLAMA_BASE_URL", "http://vllm:8000/v1")),
        )
    )
    embed_api_key = getenv("MEM0_EMBED_API_KEY", getenv("VLLM_API_KEY", "EMPTY"))

    llm_config: dict[str, Any] = {
        "model": llm_model,
        "temperature": 0,
    }
    if llm_provider == "openai":
        llm_config["openai_base_url"] = llm_base_url
        llm_config["api_key"] = llm_api_key
    else:
        llm_config["ollama_base_url"] = llm_base_url

    embed_config: dict[str, Any] = {
        "model": embed_model,
    }
    if embed_provider == "openai":
        embed_config["openai_base_url"] = embed_base_url
        embed_config["api_key"] = embed_api_key
    else:
        embed_config["ollama_base_url"] = embed_base_url

    vector_store_config: dict[str, Any] = {
        "host": getenv("MEM0_QDRANT_HOST", "qdrant"),
        "port": int(getenv("MEM0_QDRANT_PORT", "6333")),
        "collection_name": getenv("MEM0_COLLECTION_NAME", "mem0_768_v4"),
    }
    embed_dims_raw = getenv("MEM0_EMBED_DIMS", "").strip()
    allow_embed_dims_override = getenv("MEM0_EMBED_ALLOW_DIMENSIONS_OVERRIDE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if embed_dims_raw:
        try:
            embed_dims = int(embed_dims_raw)
            if embed_dims > 0:
                if embed_provider == "openai" and not allow_embed_dims_override:
                    vector_store_config["embedding_model_dims"] = embed_dims
                    embed_config["embedding_dims"] = None
                    logger.info(
                        "Ignoring embedder-level MEM0_EMBED_DIMS for OpenAI-compatible embedder; "
                        "keeping vector-store dims override and set "
                        "MEM0_EMBED_ALLOW_DIMENSIONS_OVERRIDE=1 to also force the embed request."
                    )
                else:
                    if embed_provider == "openai":
                        embed_config["embedding_dims"] = embed_dims
                    vector_store_config["embedding_model_dims"] = embed_dims
        except Exception:
            logger.warning("Invalid MEM0_EMBED_DIMS=%r; ignoring.", embed_dims_raw)

    if embed_provider == "openai" and not allow_embed_dims_override and "embedding_dims" not in embed_config:
        embed_config["embedding_dims"] = None

    clear_openai_embed_dims = embed_provider == "openai" and not allow_embed_dims_override

    config: dict[str, Any] = {
        "version": "v1.1",
        "llm": {
            "provider": llm_provider,
            "config": llm_config,
        },
        "embedder": {
            "provider": embed_provider,
            "config": embed_config,
        },
        "vector_store": {
            "provider": "qdrant",
            "config": vector_store_config,
        },
    }
    return config, clear_openai_embed_dims


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


def get_mem0_oss():
    """Create a local OSS Mem0 SDK client once. Returns None when unavailable/misconfigured."""
    global _MEM0_OSS, _MEM0_OSS_INIT_ATTEMPTED
    if _MEM0_OSS_INIT_ATTEMPTED:
        return _MEM0_OSS
    _MEM0_OSS_INIT_ATTEMPTED = True

    try:
        from mem0 import Memory  # type: ignore

        config, clear_openai_embed_dims = _build_mem0_oss_config_dict()
        _MEM0_OSS = Memory.from_config(config)  # type: ignore[attr-defined]
        if clear_openai_embed_dims:
            _clear_mem0_embedding_dims_override(_MEM0_OSS)
        return _MEM0_OSS
    except Exception as exc:
        logger.warning("Failed to initialize OSS Mem0 client; using local memory. Error: %s", exc)
        return None


async def get_mem0_oss_async():
    """Async local OSS Mem0 SDK client singleton."""
    global _MEM0_OSS_ASYNC, _MEM0_OSS_ASYNC_INIT_ATTEMPTED
    async with _get_mem0_oss_async_lock():
        if _MEM0_OSS_ASYNC_INIT_ATTEMPTED:
            return _MEM0_OSS_ASYNC
        _MEM0_OSS_ASYNC_INIT_ATTEMPTED = True

        try:
            from mem0 import AsyncMemory  # type: ignore

            config, clear_openai_embed_dims = _build_mem0_oss_config_dict()
            _MEM0_OSS_ASYNC = await AsyncMemory.from_config(config)  # type: ignore[attr-defined]
            if clear_openai_embed_dims:
                _clear_mem0_embedding_dims_override(_MEM0_OSS_ASYNC)
            return _MEM0_OSS_ASYNC
        except Exception as exc:
            logger.warning("Failed to initialize async OSS Mem0 client; using local memory. Error: %s", exc)
            return None


__all__ = [
    "get_mem0_client",
    "get_mem0_client_async",
    "get_mem0_oss",
    "get_mem0_oss_async",
    "_build_mem0_oss_config_dict",
    "_normalize_openai_base_url",
    "_clear_mem0_embedding_dims_override",
]
