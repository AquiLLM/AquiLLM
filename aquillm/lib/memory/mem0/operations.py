from __future__ import annotations

import asyncio
import inspect
import structlog
from os import getenv
import time
from typing import Any, Optional

from ..config import MEM0_TIMEOUT_SECONDS
from ..extraction import clean_stable_facts
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


def _graph_search_timeout_seconds() -> float:
    """Use a shorter timeout for graph-enabled async search before vector fallback."""
    raw = getenv("MEM0_GRAPH_SEARCH_TIMEOUT_SECONDS")
    if raw is not None and raw.strip():
        try:
            return max(1.0, min(float(MEM0_TIMEOUT_SECONDS), float(raw)))
        except ValueError:
            pass
    return max(1.0, min(float(MEM0_TIMEOUT_SECONDS), float(MEM0_TIMEOUT_SECONDS) / 3.0))


def _search_mode_label(enable_graph: Optional[bool]) -> str:
    if enable_graph is True:
        return "graph"
    if enable_graph is False:
        return "vector_only"
    return "default"


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


async def _await_result(result: Any) -> Any:
    return await result


def _run_mem0_search_call(search_callable: Any, *args: Any, **kwargs: Any) -> Any:
    """Run a Mem0 search callable in a worker thread, awaiting coroutine results there."""
    mode = kwargs.pop("_mode", "default")
    attempt = kwargs.pop("_attempt", 0)
    query_chars = kwargs.pop("_query_chars", 0)
    top_k = kwargs.pop("_top_k", 0)
    started_at = time.perf_counter()
    try:
        result = search_callable(*args, **kwargs)
        if inspect.isawaitable(result):
            resolved = asyncio.run(_await_result(result))
        else:
            resolved = result
        logger.info(
            "Mem0 raw search call completed: mode=%s attempt=%d elapsed_ms=%.2f "
            "query_chars=%d top_k=%d",
            mode,
            attempt,
            (time.perf_counter() - started_at) * 1000.0,
            query_chars,
            top_k,
        )
        return resolved
    except TypeError:
        raise
    except Exception as exc:
        logger.warning(
            "Mem0 raw search call failed: mode=%s attempt=%d elapsed_ms=%.2f "
            "query_chars=%d top_k=%d error_type=%s error=%s",
            mode,
            attempt,
            (time.perf_counter() - started_at) * 1000.0,
            query_chars,
            top_k,
            type(exc).__name__,
            exc,
        )
        raise


async def _call_mem0_search_async(
    mem0: Any, query: str, user_id: str, top_k: int, enable_graph: Optional[bool]
) -> Any:
    attempts = _build_search_call_attempts(query, user_id, top_k, enable_graph=enable_graph)
    timeout = _graph_search_timeout_seconds() if enable_graph else float(MEM0_TIMEOUT_SECONDS)
    mode = _search_mode_label(enable_graph)
    query_chars = len(query or "")
    for attempt_number, (args, kwargs) in enumerate(attempts, start=1):
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(
                    _run_mem0_search_call,
                    mem0.search,
                    *args,
                    **(
                        kwargs
                        | {
                            "_mode": mode,
                            "_attempt": attempt_number,
                            "_query_chars": query_chars,
                            "_top_k": top_k,
                        }
                    ),
                ),  # type: ignore[attr-defined]
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Mem0 raw search attempt timed out: mode=%s attempt=%d timeout_s=%.1f "
                "query_chars=%d top_k=%d",
                mode,
                attempt_number,
                timeout,
                query_chars,
                top_k,
            )
            raise
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
    graph_timeout_seconds = _graph_search_timeout_seconds() if enable_graph else float(MEM0_TIMEOUT_SECONDS)
    graph_started_at = time.perf_counter()
    try:
        response = await _call_mem0_search_async(mem0, query, user_id, top_k, enable_graph=enable_graph)
        if enable_graph:
            logger.info(
                "Mem0 OSS async graph search completed: graph_timeout_s=%.1f overall_timeout_s=%.1f "
                "elapsed_ms=%.2f query_chars=%d top_k=%d",
                graph_timeout_seconds,
                float(MEM0_TIMEOUT_SECONDS),
                (time.perf_counter() - graph_started_at) * 1000.0,
                len(query or ""),
                top_k,
            )
    except asyncio.TimeoutError:
        if enable_graph and _graph_fail_open():
            logger.warning(
                "Mem0 OSS async graph search timed out after %.1fs; retrying vector-only "
                "(overall timeout %.1fs).",
                graph_timeout_seconds,
                float(MEM0_TIMEOUT_SECONDS),
            )
            try:
                retry_started_at = time.perf_counter()
                response = await _call_mem0_search_async(mem0, query, user_id, top_k, enable_graph=False)
                logger.info(
                    "Mem0 OSS async vector-only retry completed after graph timeout: "
                    "elapsed_ms=%.2f query_chars=%d top_k=%d",
                    (time.perf_counter() - retry_started_at) * 1000.0,
                    len(query or ""),
                    top_k,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Mem0 OSS async vector-only retry timed out after %.1fs; falling back.",
                    float(MEM0_TIMEOUT_SECONDS),
                )
                return []
            except Exception as retry_exc:
                logger.warning("Mem0 OSS async vector-only retry failed; falling back. Error: %s", retry_exc)
                return []
        else:
            logger.warning("Mem0 OSS async search timed out after %ss", MEM0_TIMEOUT_SECONDS)
            return []
    except Exception as exc:
        if enable_graph and _graph_fail_open():
            logger.warning(
                "Mem0 OSS async graph search failed after %.2fms; retrying vector-only. Error: %s",
                (time.perf_counter() - graph_started_at) * 1000.0,
                exc,
            )
            try:
                retry_started_at = time.perf_counter()
                response = await _call_mem0_search_async(mem0, query, user_id, top_k, enable_graph=False)
                logger.info(
                    "Mem0 OSS async vector-only retry completed after graph failure: "
                    "elapsed_ms=%.2f query_chars=%d top_k=%d",
                    (time.perf_counter() - retry_started_at) * 1000.0,
                    len(query or ""),
                    top_k,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Mem0 OSS async vector-only retry timed out after %.1fs; falling back.",
                    float(MEM0_TIMEOUT_SECONDS),
                )
                return []
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
    cleaned_facts = clean_stable_facts(facts)
    filtered_count = max(len(facts) - len(cleaned_facts), 0)
    logger.info(
        "Mem0 raw fact candidates prepared: extracted=%d filtered=%d graph_enabled=%s",
        len(facts),
        filtered_count,
        enable_graph,
    )
    if not cleaned_facts:
        logger.info("Mem0 raw fact candidates filtered before write; skipping add call.")
        return False
    for fact in cleaned_facts:
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
    cleaned_facts = clean_stable_facts(facts)
    filtered_count = max(len(facts) - len(cleaned_facts), 0)
    logger.info(
        "Mem0 raw fact candidates prepared (async): extracted=%d filtered=%d graph_enabled=%s",
        len(facts),
        filtered_count,
        enable_graph,
    )
    if not cleaned_facts:
        logger.info("Mem0 raw fact candidates filtered before async write; skipping add call.")
        return False
    for fact in cleaned_facts:
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
