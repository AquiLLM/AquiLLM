from __future__ import annotations

import asyncio
import structlog
from os import getenv
from typing import Any, Optional

from ..config import MEM0_TIMEOUT_SECONDS
from ..types import RetrievedEpisodicMemory
from .client import get_mem0_oss, get_mem0_oss_async
from .search_parsing import parse_mem0_search_items, response_to_raw_items

logger = structlog.stdlib.get_logger(__name__)

_NO_RESPONSE = object()
def _env_bool(name: str, default: bool) -> bool:
    raw = getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _graph_fail_open() -> bool:
    return _env_bool("MEM0_GRAPH_FAIL_OPEN", True)


def _search_enable_graph() -> bool:
    return _env_bool("MEM0_GRAPH_ENABLED", False) and _env_bool("MEM0_GRAPH_SEARCH_ENABLED", True)


def _add_enable_graph() -> bool:
    return _env_bool("MEM0_GRAPH_ENABLED", False) and _env_bool("MEM0_GRAPH_ADD_ENABLED", True)
def _build_search_call_attempts(
    query: str, user_id: str, top_k: int, enable_graph: Optional[bool]
) -> list[tuple[tuple[Any, ...], dict[str, Any]]]:
    search_kwargs: dict[str, Any] = {}
    if enable_graph is not None:
        search_kwargs["enable_graph"] = enable_graph
    return [
        ((), {"query": query, "user_id": user_id, "limit": top_k, **search_kwargs}),
        ((), {"query": query, "user_id": user_id, "top_k": top_k, **search_kwargs}),
        ((), {"query": query, "user_id": user_id, **search_kwargs}),
        ((query,), {"user_id": user_id, "limit": top_k, **search_kwargs}),
        ((query,), {"user_id": user_id, **search_kwargs}),
    ]


def _call_mem0_search_sync(
    mem0: Any, query: str, user_id: str, top_k: int, enable_graph: Optional[bool]
) -> Any:
    attempts = _build_search_call_attempts(query, user_id, top_k, enable_graph=enable_graph)
    for args, kwargs in attempts:
        try:
            return mem0.search(*args, **kwargs)  # type: ignore[attr-defined]
        except TypeError:
            continue
    if enable_graph is not None:
        return _call_mem0_search_sync(mem0, query, user_id, top_k, enable_graph=None)
    return _NO_RESPONSE
async def _call_mem0_search_async(
    mem0: Any, query: str, user_id: str, top_k: int, enable_graph: Optional[bool]
) -> Any:
    attempts = _build_search_call_attempts(query, user_id, top_k, enable_graph=enable_graph)
    for args, kwargs in attempts:
        try:
            return await asyncio.wait_for(
                mem0.search(*args, **kwargs),  # type: ignore[attr-defined]
                timeout=float(MEM0_TIMEOUT_SECONDS),
            )
        except TypeError:
            continue
    if enable_graph is not None:
        return await _call_mem0_search_async(mem0, query, user_id, top_k, enable_graph=None)
    return _NO_RESPONSE
def search_mem0_via_oss(
    user_id: str, query: str, top_k: int, exclude_conversation_id: Optional[int]
) -> list[RetrievedEpisodicMemory]:
    mem0 = get_mem0_oss()
    if mem0 is None:
        return []

    enable_graph = _search_enable_graph()
    try:
        response = _call_mem0_search_sync(mem0, query, user_id, top_k, enable_graph=enable_graph)
    except Exception as exc:
        if enable_graph and _graph_fail_open():
            logger.warning("Mem0 OSS graph search failed; retrying vector-only. Error: %s", exc)
            try:
                response = _call_mem0_search_sync(mem0, query, user_id, top_k, enable_graph=False)
            except Exception as retry_exc:
                logger.warning("Mem0 OSS vector-only retry failed; falling back. Error: %s", retry_exc)
                return []
        else:
            logger.warning("Mem0 OSS search failed; falling back. Error: %s", exc)
            return []

    if response is _NO_RESPONSE:
        logger.warning("Mem0 OSS search call signature not supported; falling back.")
        return []

    return parse_mem0_search_items(
        raw_items=response_to_raw_items(response),
        top_k=top_k,
        exclude_conversation_id=exclude_conversation_id,
    )
async def search_mem0_via_oss_async(
    user_id: str, query: str, top_k: int, exclude_conversation_id: Optional[int]
) -> list[RetrievedEpisodicMemory]:
    """Async variant of search_mem0_via_oss using AsyncMemory."""
    mem0 = await get_mem0_oss_async()
    if mem0 is None:
        return []

    enable_graph = _search_enable_graph()
    try:
        response = await _call_mem0_search_async(mem0, query, user_id, top_k, enable_graph=enable_graph)
    except asyncio.TimeoutError:
        if enable_graph and _graph_fail_open():
            logger.warning(
                "Mem0 OSS async graph search timed out after %ss; retrying vector-only.",
                MEM0_TIMEOUT_SECONDS,
            )
            try:
                response = await _call_mem0_search_async(mem0, query, user_id, top_k, enable_graph=False)
            except Exception as retry_exc:
                logger.warning("Mem0 OSS async vector-only retry failed; falling back. Error: %s", retry_exc)
                return []
        else:
            logger.warning("Mem0 OSS async search timed out after %ss", MEM0_TIMEOUT_SECONDS)
            return []
    except Exception as exc:
        if enable_graph and _graph_fail_open():
            logger.warning("Mem0 OSS async graph search failed; retrying vector-only. Error: %s", exc)
            try:
                response = await _call_mem0_search_async(mem0, query, user_id, top_k, enable_graph=False)
            except Exception as retry_exc:
                logger.warning("Mem0 OSS async vector-only retry failed; falling back. Error: %s", retry_exc)
                return []
        else:
            logger.warning("Mem0 OSS async search failed; falling back. Error: %s", exc)
            return []

    if response is _NO_RESPONSE:
        logger.warning("Mem0 OSS async search call signature not supported; falling back.")
        return []

    return parse_mem0_search_items(
        raw_items=response_to_raw_items(response),
        top_k=top_k,
        exclude_conversation_id=exclude_conversation_id,
    )
def search_mem0_episodic_memories(
    user_id: str, query: str, top_k: int, exclude_conversation_id: Optional[int]
) -> list[RetrievedEpisodicMemory]:
    return search_mem0_via_oss(user_id=user_id, query=query, top_k=top_k, exclude_conversation_id=exclude_conversation_id)
async def search_mem0_episodic_memories_async(
    user_id: str, query: str, top_k: int, exclude_conversation_id: Optional[int]
) -> list[RetrievedEpisodicMemory]:
    return await search_mem0_via_oss_async(
        user_id=user_id, query=query, top_k=top_k, exclude_conversation_id=exclude_conversation_id
    )
def _add_mem0_fact(mem0: Any, fact: str, user_id: str, metadata: dict[str, Any], enable_graph: Optional[bool]) -> Any:
    add_kwargs = {"user_id": user_id, "metadata": metadata, "infer": False}
    if enable_graph is not None:
        add_kwargs["enable_graph"] = enable_graph
    try:
        return mem0.add(fact, **add_kwargs)  # type: ignore[attr-defined]
    except TypeError:
        if enable_graph is None:
            raise
        return _add_mem0_fact(mem0, fact, user_id, metadata, enable_graph=None)
async def _add_mem0_fact_async(
    mem0: Any, fact: str, user_id: str, metadata: dict[str, Any], enable_graph: Optional[bool]
) -> Any:
    add_kwargs = {"user_id": user_id, "metadata": metadata, "infer": False}
    if enable_graph is not None:
        add_kwargs["enable_graph"] = enable_graph
    try:
        return await asyncio.wait_for(
            mem0.add(fact, **add_kwargs),  # type: ignore[attr-defined]
            timeout=float(MEM0_TIMEOUT_SECONDS),
        )
    except TypeError:
        if enable_graph is None:
            raise
        return await _add_mem0_fact_async(mem0, fact, user_id, metadata, enable_graph=None)
def add_mem0_raw_facts(
    user_id: str,
    facts: list[str],
    conversation_id: int,
    assistant_message_uuid: str,
) -> bool:
    mem0 = get_mem0_oss()
    if mem0 is None:
        return False

    metadata = {
        "conversation_id": conversation_id,
        "assistant_message_uuid": assistant_message_uuid,
        "source": "aquillm",
        "memory_type": "episodic",
    }

    wrote_any = False
    enable_graph = _add_enable_graph()
    for fact in facts:
        try:
            result = _add_mem0_fact(mem0, fact, user_id, metadata, enable_graph=enable_graph)
        except Exception as exc:
            if enable_graph and _graph_fail_open():
                logger.warning("Mem0 graph add failed for fact=%r; retrying vector-only. Error: %s", fact, exc)
                try:
                    result = _add_mem0_fact(mem0, fact, user_id, metadata, enable_graph=False)
                except Exception as retry_exc:
                    logger.warning("Mem0 vector-only add retry failed for fact=%r: %s", fact, retry_exc)
                    continue
            else:
                logger.warning("Mem0 raw fact add failed for fact=%r: %s", fact, exc)
                continue

        if isinstance(result, dict):
            events = result.get("results") or []
            if any(isinstance(x, dict) and x.get("event") == "ADD" for x in events):
                wrote_any = True
        else:
            wrote_any = True
    return wrote_any
async def add_mem0_raw_facts_async(
    user_id: str,
    facts: list[str],
    conversation_id: int,
    assistant_message_uuid: str,
) -> bool:
    mem0 = await get_mem0_oss_async()
    if mem0 is None:
        return False
    metadata = {
        "conversation_id": conversation_id,
        "assistant_message_uuid": assistant_message_uuid,
        "source": "aquillm",
        "memory_type": "episodic",
    }
    wrote_any = False
    enable_graph = _add_enable_graph()
    for fact in facts:
        try:
            result = await _add_mem0_fact_async(mem0, fact, user_id, metadata, enable_graph=enable_graph)
        except Exception as exc:
            if not (enable_graph and _graph_fail_open()):
                logger.warning("Mem0 raw fact add failed for fact=%r: %s", fact, exc)
                continue
            logger.warning("Mem0 graph add failed for fact=%r; retrying vector-only. Error: %s", fact, exc)
            try:
                result = await _add_mem0_fact_async(mem0, fact, user_id, metadata, enable_graph=False)
            except Exception as retry_exc:
                logger.warning("Mem0 vector-only add retry failed for fact=%r: %s", fact, retry_exc)
                continue
        if isinstance(result, dict):
            events = result.get("results") or []
            wrote_any = wrote_any or any(isinstance(x, dict) and x.get("event") == "ADD" for x in events)
        else:
            wrote_any = True
    return wrote_any
def add_mem0_memory_with_client(
    user_id: str,
    user_content: str,
    assistant_content: str,
    conversation_id: int,
    assistant_message_uuid: str,
) -> None:
    logger.info("Mem0 cloud fallback disabled; skipping cloud add for user_id=%s", user_id)


__all__ = [
    "search_mem0_episodic_memories",
    "search_mem0_episodic_memories_async",
    "search_mem0_via_oss",
    "search_mem0_via_oss_async",
    "add_mem0_raw_facts",
    "add_mem0_raw_facts_async",
    "add_mem0_memory_with_client",
]
