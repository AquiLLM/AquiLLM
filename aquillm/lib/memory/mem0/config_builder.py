"""Mem0 OSS configuration and compatibility helpers."""

from __future__ import annotations

from os import getenv
from typing import Any, Optional

import structlog

logger = structlog.stdlib.get_logger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    raw = getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_optional_str(name: str) -> str | None:
    value = (getenv(name, "") or "").strip()
    return value or None


def _env_optional_float(name: str) -> float | None:
    value = _env_optional_str(name)
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        logger.warning("Invalid %s=%r; ignoring.", name, value)
        return None


def normalize_openai_base_url(url: str) -> str:
    """Normalize OpenAI-compatible base URL to include /v1 suffix."""
    base = (url or "").strip().rstrip("/")
    if not base:
        return "http://vllm:8000/v1"
    if base.endswith("/v1"):
        return base
    return f"{base}/v1"


def clear_mem0_embedding_dims_override(value: Any, seen: Optional[set[int]] = None) -> None:
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
            clear_mem0_embedding_dims_override(child, seen)
        return

    if isinstance(value, (list, tuple, set)):
        for child in value:
            clear_mem0_embedding_dims_override(child, seen)
        return

    if hasattr(value, "embedding_dims"):
        try:
            setattr(value, "embedding_dims", None)
        except Exception:
            pass

    child_dict = getattr(value, "__dict__", None)
    if isinstance(child_dict, dict):
        clear_mem0_embedding_dims_override(child_dict, seen)


def _build_mem0_graph_store() -> dict[str, Any] | None:
    graph_enabled = _env_bool("MEM0_GRAPH_ENABLED", False)
    fail_open = _env_bool("MEM0_GRAPH_FAIL_OPEN", True)
    add_enabled = _env_bool("MEM0_GRAPH_ADD_ENABLED", True)
    search_enabled = _env_bool("MEM0_GRAPH_SEARCH_ENABLED", True)
    provider = (getenv("MEM0_GRAPH_PROVIDER", "") or "").strip().lower()
    url = (getenv("MEM0_GRAPH_URL", "bolt://memgraph:7687") or "").strip()
    username = (getenv("MEM0_GRAPH_USERNAME", "memgraph") or "").strip()
    password = (getenv("MEM0_GRAPH_PASSWORD", "") or "").strip()
    database = _env_optional_str("MEM0_GRAPH_DATABASE")
    custom_prompt = _env_optional_str("MEM0_GRAPH_CUSTOM_PROMPT")
    threshold = _env_optional_float("MEM0_GRAPH_THRESHOLD")

    if not graph_enabled:
        logger.info("Mem0 graph mode disabled; using vector-only memory.")
        return None

    if not provider:
        message = "MEM0_GRAPH_ENABLED=1 requires MEM0_GRAPH_PROVIDER."
        if fail_open:
            logger.warning("%s Fail-open enabled; continuing vector-only.", message)
            return None
        raise ValueError(message)
    if provider != "memgraph":
        message = f"Unsupported MEM0 graph provider: {provider!r}."
        if fail_open:
            logger.warning("%s Fail-open enabled; continuing vector-only.", message)
            return None
        raise ValueError(message)

    missing: list[str] = []
    if not url:
        missing.append("MEM0_GRAPH_URL")
    if not username:
        missing.append("MEM0_GRAPH_USERNAME")
    if not password:
        missing.append("MEM0_GRAPH_PASSWORD")
    if missing:
        message = f"Incomplete MEM0 graph config; missing {', '.join(missing)}."
        if fail_open:
            logger.warning("%s Fail-open enabled; continuing vector-only.", message)
            return None
        raise ValueError(message)

    graph_config: dict[str, Any] = {
        "url": url,
        "username": username,
        "password": password,
        # Mem0's Memgraph adapter initializes langchain-memgraph under the hood.
        # Explicitly disable schema refresh because its bootstrap query has been
        # incompatible with the Memgraph version we run in Docker.
        "refresh_schema": False,
    }
    if database:
        graph_config["database"] = database
    if threshold is not None:
        graph_config["threshold"] = threshold

    logger.info(
        "Mem0 graph mode enabled: provider=%s add_enabled=%s search_enabled=%s",
        provider,
        add_enabled,
        search_enabled,
    )
    graph_store = {"provider": provider, "config": graph_config}
    if custom_prompt:
        graph_store["custom_prompt"] = custom_prompt
    return graph_store


def build_mem0_oss_config_dict() -> tuple[dict[str, Any], bool]:
    """
    Build Mem0 OSS SDK config dict for Memory / AsyncMemory.

    Returns:
        (config_dict, clear_openai_embed_dims) - when True, run clear_mem0_embedding_dims_override on the client.
    """
    llm_provider = getenv("MEM0_LLM_PROVIDER", "openai")
    llm_model = getenv("MEM0_LLM_MODEL", "qwen3.5:4b-q8_0")
    llm_base_url = normalize_openai_base_url(
        getenv(
            "MEM0_LLM_BASE_URL",
            getenv("MEM0_VLLM_BASE_URL", getenv("MEM0_OLLAMA_BASE_URL", "http://vllm:8000/v1")),
        )
    )
    llm_api_key = getenv("MEM0_LLM_API_KEY", getenv("VLLM_API_KEY", "EMPTY"))

    embed_provider = getenv("MEM0_EMBED_PROVIDER", "openai")
    embed_model = getenv("MEM0_EMBED_MODEL", "Qwen/Qwen3-Embedding-4B")
    embed_base_url = normalize_openai_base_url(
        getenv(
            "MEM0_EMBED_BASE_URL",
            getenv("MEM0_VLLM_BASE_URL", getenv("MEM0_OLLAMA_BASE_URL", "http://vllm:8000/v1")),
        )
    )
    embed_api_key = getenv("MEM0_EMBED_API_KEY", getenv("VLLM_API_KEY", "EMPTY"))

    llm_config: dict[str, Any] = {"model": llm_model, "temperature": 0}
    if llm_provider == "openai":
        llm_config["openai_base_url"] = llm_base_url
        llm_config["api_key"] = llm_api_key
    else:
        llm_config["ollama_base_url"] = llm_base_url

    embed_config: dict[str, Any] = {"model": embed_model}
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
        "llm": {"provider": llm_provider, "config": llm_config},
        "embedder": {"provider": embed_provider, "config": embed_config},
        "vector_store": {"provider": "qdrant", "config": vector_store_config},
    }
    graph_store = _build_mem0_graph_store()
    if graph_store is not None:
        config["graph_store"] = graph_store
    return config, clear_openai_embed_dims


__all__ = [
    "build_mem0_oss_config_dict",
    "clear_mem0_embedding_dims_override",
    "normalize_openai_base_url",
]
