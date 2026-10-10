from __future__ import annotations

import asyncio
import inspect
import queue
import structlog
import threading
from os import getenv
import time
from typing import Any, Optional

from ..config import MEM0_TIMEOUT_SECONDS
from ..extraction import clean_stable_facts
from ..types import RetrievedEpisodicMemory
from .client import get_mem0_oss, get_mem0_oss_async, get_mem0_oss_async_vector, get_mem0_oss_vector
from .search_parsing import parse_mem0_search_items, response_to_raw_items

from ._write_payload import (
    _payload_shape_error, _messages_to_transcript, _message_payload_candidates,
    _mem0_add_result_has_writes, _add_mem0_messages_payload as _write_messages_payload,
)

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


def _build_search_call_attempts(query: str, user_id: str, top_k: int) -> list[tuple[tuple[Any, ...], dict[str, Any]]]:
    return [
        ((), {"query": query, "user_id": user_id, "limit": top_k}),
        ((), {"query": query, "user_id": user_id, "top_k": top_k}),
        ((), {"query": query, "user_id": user_id}),
        ((query,), {"user_id": user_id, "limit": top_k}),
        ((query,), {"user_id": user_id}),
    ]


def _call_mem0_search_sync(mem0: Any, query: str, user_id: str, top_k: int) -> Any:
    attempts = _build_search_call_attempts(query, user_id, top_k)
    for args, kwargs in attempts:
        try:
            return mem0.search(*args, **kwargs)  # type: ignore[attr-defined]
        except TypeError:
            continue
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
            "obs.memory.raw_search_completed",
            mode=mode,
            attempt=attempt,
            elapsed_ms=(time.perf_counter() - started_at) * 1000.0,
            query_chars=query_chars,
            top_k=top_k,
        )
        return resolved
    except TypeError:
        raise
    except Exception as exc:
        logger.warning(
            "obs.memory.raw_search_error",
            mode=mode,
            attempt=attempt,
            elapsed_ms=(time.perf_counter() - started_at) * 1000.0,
            query_chars=query_chars,
            top_k=top_k,
            error=str(exc),
            error_type=type(exc).__name__,
        )
        raise


async def _run_mem0_search_call_async(search_callable: Any, *args: Any, **kwargs: Any) -> Any:
    """Run an async Mem0 search callable directly so timeouts can cancel it."""
    mode = kwargs.pop("_mode", "default")
    attempt = kwargs.pop("_attempt", 0)
    query_chars = kwargs.pop("_query_chars", 0)
    top_k = kwargs.pop("_top_k", 0)
    started_at = time.perf_counter()
    try:
        resolved = await _await_result(search_callable(*args, **kwargs))
        logger.info(
            "obs.memory.raw_search_completed",
            mode=mode,
            attempt=attempt,
            elapsed_ms=(time.perf_counter() - started_at) * 1000.0,
            query_chars=query_chars,
            top_k=top_k,
        )
        return resolved
    except TypeError:
        raise
    except Exception as exc:
        logger.warning(
            "obs.memory.raw_search_error",
            mode=mode,
            attempt=attempt,
            elapsed_ms=(time.perf_counter() - started_at) * 1000.0,
            query_chars=query_chars,
            top_k=top_k,
            error=str(exc),
            error_type=type(exc).__name__,
        )
        raise


async def _call_mem0_search_async(
    mem0: Any, query: str, user_id: str, top_k: int, enable_graph: Optional[bool]
) -> Any:
    attempts = _build_search_call_attempts(query, user_id, top_k)
    timeout = _graph_search_timeout_seconds() if enable_graph else float(MEM0_TIMEOUT_SECONDS)
    mode = _search_mode_label(enable_graph)
    query_chars = len(query or "")
    search_callable = mem0.search  # type: ignore[attr-defined]
    is_async_search = inspect.iscoroutinefunction(search_callable)
    for attempt_number, (args, kwargs) in enumerate(attempts, start=1):
        try:
            call_kwargs = kwargs | {
                "_mode": mode,
                "_attempt": attempt_number,
                "_query_chars": query_chars,
                "_top_k": top_k,
            }
            if is_async_search:
                return await asyncio.wait_for(
                    _run_mem0_search_call_async(search_callable, *args, **call_kwargs),
                    timeout=timeout,
                )
            return await asyncio.wait_for(
                asyncio.to_thread(
                    _run_mem0_search_call,
                    search_callable,
                    *args,
                    **call_kwargs,
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "obs.memory.raw_search_timeout",
                mode=mode,
                attempt=attempt_number,
                timeout_s=timeout,
                query_chars=query_chars,
                top_k=top_k,
            )
            raise
        except TypeError:
            continue
    return _NO_RESPONSE
def search_mem0_via_oss(
    user_id: str, query: str, top_k: int, exclude_conversation_id: Optional[int]
) -> list[RetrievedEpisodicMemory]:
    enable_graph = _search_enable_graph()
    mem0 = get_mem0_oss() if enable_graph else get_mem0_oss_vector()
    if mem0 is None:
        return []

    try:
        response = _call_mem0_search_sync(mem0, query, user_id, top_k)
    except Exception as exc:
        if enable_graph and _graph_fail_open():
            logger.warning(
                "obs.memory.graph_search_error",
                error=str(exc),
                error_type=type(exc).__name__,
            )
            try:
                vector_mem0 = get_mem0_oss_vector()
                if vector_mem0 is None:
                    return []
                response = _call_mem0_search_sync(vector_mem0, query, user_id, top_k)
            except Exception as retry_exc:
                logger.warning(
                    "obs.memory.vector_retry_error",
                    error=str(retry_exc),
                    error_type=type(retry_exc).__name__,
                )
                return []
        else:
            logger.warning(
                "obs.memory.search_error",
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return []

    if response is _NO_RESPONSE:
        logger.warning("obs.memory.search_signature_unsupported")
        return []

    return parse_mem0_search_items(
        raw_items=response_to_raw_items(response),
        top_k=top_k,
        exclude_conversation_id=exclude_conversation_id,
        user_id=user_id,
    )
async def search_mem0_via_oss_async(
    user_id: str, query: str, top_k: int, exclude_conversation_id: Optional[int]
) -> list[RetrievedEpisodicMemory]:
    """Async variant of search_mem0_via_oss using AsyncMemory."""
    enable_graph = _search_enable_graph()
    mem0 = await (get_mem0_oss_async() if enable_graph else get_mem0_oss_async_vector())
    if mem0 is None:
        return []

    graph_timeout_seconds = _graph_search_timeout_seconds() if enable_graph else float(MEM0_TIMEOUT_SECONDS)
    graph_started_at = time.perf_counter()
    try:
        response = await _call_mem0_search_async(mem0, query, user_id, top_k, enable_graph=enable_graph)
        if enable_graph:
            logger.info(
                "obs.memory.async_graph_search_completed",
                graph_timeout_s=graph_timeout_seconds,
                overall_timeout_s=float(MEM0_TIMEOUT_SECONDS),
                elapsed_ms=(time.perf_counter() - graph_started_at) * 1000.0,
                query_chars=len(query or ""),
                top_k=top_k,
            )
    except asyncio.TimeoutError:
        if enable_graph and _graph_fail_open():
            logger.warning(
                "obs.memory.async_graph_search_timeout",
                graph_timeout_s=graph_timeout_seconds,
                overall_timeout_s=float(MEM0_TIMEOUT_SECONDS),
            )
            try:
                retry_started_at = time.perf_counter()
                vector_mem0 = await get_mem0_oss_async_vector()
                if vector_mem0 is None:
                    return []
                response = await _call_mem0_search_async(vector_mem0, query, user_id, top_k, enable_graph=False)
                logger.info(
                    "obs.memory.async_vector_retry_completed",
                    after="graph_timeout",
                    elapsed_ms=(time.perf_counter() - retry_started_at) * 1000.0,
                    query_chars=len(query or ""),
                    top_k=top_k,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "obs.memory.async_vector_retry_timeout",
                    overall_timeout_s=float(MEM0_TIMEOUT_SECONDS),
                )
                return []
            except Exception as retry_exc:
                logger.warning(
                    "obs.memory.async_vector_retry_error",
                    error=str(retry_exc),
                    error_type=type(retry_exc).__name__,
                )
                return []
        else:
            logger.warning(
                "obs.memory.async_search_timeout",
                timeout_s=MEM0_TIMEOUT_SECONDS,
            )
            return []
    except Exception as exc:
        if enable_graph and _graph_fail_open():
            logger.warning(
                "obs.memory.async_graph_search_error",
                elapsed_ms=(time.perf_counter() - graph_started_at) * 1000.0,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            try:
                retry_started_at = time.perf_counter()
                vector_mem0 = await get_mem0_oss_async_vector()
                if vector_mem0 is None:
                    return []
                response = await _call_mem0_search_async(vector_mem0, query, user_id, top_k, enable_graph=False)
                logger.info(
                    "obs.memory.async_vector_retry_completed",
                    after="graph_failure",
                    elapsed_ms=(time.perf_counter() - retry_started_at) * 1000.0,
                    query_chars=len(query or ""),
                    top_k=top_k,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "obs.memory.async_vector_retry_timeout",
                    overall_timeout_s=float(MEM0_TIMEOUT_SECONDS),
                )
                return []
            except Exception as retry_exc:
                logger.warning(
                    "obs.memory.async_vector_retry_error",
                    error=str(retry_exc),
                    error_type=type(retry_exc).__name__,
                )
                return []
        else:
            logger.warning(
                "obs.memory.async_search_error",
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return []

    if response is _NO_RESPONSE:
        logger.warning("obs.memory.async_search_signature_unsupported")
        return []

    return parse_mem0_search_items(
        raw_items=response_to_raw_items(response),
        top_k=top_k,
        exclude_conversation_id=exclude_conversation_id,
        user_id=user_id,
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


def _run_mem0_add_call_with_timeout(add_callable: Any, *args: Any, **kwargs: Any) -> Any:
    timeout_s = float(MEM0_TIMEOUT_SECONDS)
    result_queue: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

    def _target() -> None:
        try:
            result_queue.put((True, add_callable(*args, **kwargs)))
        except Exception as exc:
            result_queue.put((False, exc))

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout_s)
    if thread.is_alive():
        raise TimeoutError(f"Mem0 add timed out after {timeout_s:.1f}s")

    ok, value = result_queue.get_nowait()
    if ok:
        return value
    raise value


def _add_mem0_fact(mem0: Any, fact: str, user_id: str, metadata: dict[str, Any], enable_graph: Optional[bool]) -> Any:
    add_kwargs = {"user_id": user_id, "metadata": metadata, "infer": False}
    if enable_graph is not None:
        add_kwargs["enable_graph"] = enable_graph
    try:
        return _run_mem0_add_call_with_timeout(mem0.add, fact, **add_kwargs)  # type: ignore[attr-defined]
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


def _add_mem0_messages_payload(mem0, messages, user_id, metadata, enable_graph, *, strict=False):
    return _write_messages_payload(mem0, messages, user_id, metadata, enable_graph,
                                   strict=strict, timeout_call=_run_mem0_add_call_with_timeout)


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
        "user_id": user_id,
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
        "obs.memory.raw_facts_prepared",
        extracted_count=len(facts),
        filtered_count=filtered_count,
        graph_enabled=enable_graph,
    )
    if not cleaned_facts:
        logger.info("obs.memory.raw_facts_skipped")
        return False
    for fact in cleaned_facts:
        try:
            result = _add_mem0_fact(mem0, fact, user_id, metadata, enable_graph=enable_graph)
        except Exception as exc:
            if enable_graph and _graph_fail_open():
                logger.warning(
                    "obs.memory.graph_add_error",
                    fact=fact,
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
                try:
                    result = _add_mem0_fact(mem0, fact, user_id, metadata, enable_graph=False)
                except Exception as retry_exc:
                    logger.warning(
                        "obs.memory.vector_add_retry_error",
                        fact=fact,
                        error=str(retry_exc),
                        error_type=type(retry_exc).__name__,
                    )
                    continue
            else:
                logger.warning(
                    "obs.memory.raw_fact_add_error",
                    fact=fact,
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
                continue

        if isinstance(result, dict):
            events = result.get("results") or []
            if any(isinstance(x, dict) and x.get("event") == "ADD" for x in events):
                wrote_any = True
        else:
            wrote_any = True
    return wrote_any


def add_mem0_messages(
    user_id: str,
    messages: list[dict[str, Any]],
    conversation_id: int,
    assistant_message_uuid: str,
    *, strict: bool = False,
    source_message_uuid=None, source_content_digest=None, result_observer=None, coverage_observer=None,
) -> bool:
    """Return whether inference added memories; strict callers receive failures.

    An empty successful result is False in either mode. Strict mode waits for
    the SDK call itself, whose network timeouts remain authoritative, so the
    durable worker can retain execution ownership until the call ends.
    """
    from ..extraction import parse_memory_intent
    user_intents = [(item, parse_memory_intent(item.get('content', '')))
                    for item in messages if isinstance(item, dict) and item.get('role') == 'user']
    if any(intent.kind not in {'none', 'affirmative'} for _, intent in user_intents):
        return False
    if any(intent.kind == 'affirmative' for _, intent in user_intents):
        messages = [{'role': 'user', 'content': intent.payload}
                    for _, intent in user_intents if intent.kind == 'affirmative']
    mem0 = get_mem0_oss()
    if mem0 is None:
        if strict:
            raise RuntimeError('Mem0 client is unavailable')
        return False
    if not messages:
        logger.info("obs.memory.intelligent_write_skipped")
        return False

    metadata = {
        "user_id": user_id,
        "conversation_id": conversation_id,
        "assistant_message_uuid": assistant_message_uuid,
        "source": "aquillm",
        "memory_type": "episodic",
    }

    if source_message_uuid is not None:
        metadata['source_message_uuid'] = str(source_message_uuid)
        metadata['source_content_digest'] = source_content_digest
    enable_graph = _add_enable_graph()
    try:
        result = _add_mem0_messages_payload(mem0, messages, user_id, metadata, enable_graph=enable_graph, strict=strict)
    except Exception as exc:
        if enable_graph and _graph_fail_open():
            logger.warning(
                "obs.memory.intelligent_write_graph_error",
                error_type=type(exc).__name__,
            )
            try:
                result = _add_mem0_messages_payload(mem0, messages, user_id, metadata, enable_graph=False, strict=strict)
            except Exception as retry_exc:
                logger.warning(
                    "obs.memory.intelligent_write_vector_retry_error",
                    error_type=type(retry_exc).__name__,
                )
                if strict:
                    raise
                return False
        else:
            logger.warning(
                "obs.memory.intelligent_write_error",
                error_type=type(exc).__name__,
            )
            if strict:
                raise
            return False

    if result_observer is not None:
        result_observer(result)
    if coverage_observer is not None:
        coverage_observer(enable_graph)  # Includes a failed graph attempt before vector retry.
    return _mem0_add_result_has_writes(result)


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
        "user_id": user_id,
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
        "obs.memory.raw_facts_prepared_async",
        extracted_count=len(facts),
        filtered_count=filtered_count,
        graph_enabled=enable_graph,
    )
    if not cleaned_facts:
        logger.info("obs.memory.raw_facts_skipped_async")
        return False
    for fact in cleaned_facts:
        try:
            result = await _add_mem0_fact_async(mem0, fact, user_id, metadata, enable_graph=enable_graph)
        except Exception as exc:
            if not (enable_graph and _graph_fail_open()):
                logger.warning(
                    "obs.memory.raw_fact_add_error",
                    fact=fact,
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
                continue
            logger.warning(
                "obs.memory.graph_add_error",
                fact=fact,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            try:
                result = await _add_mem0_fact_async(mem0, fact, user_id, metadata, enable_graph=False)
            except Exception as retry_exc:
                logger.warning(
                    "obs.memory.vector_add_retry_error",
                    fact=fact,
                    error=str(retry_exc),
                    error_type=type(retry_exc).__name__,
                )
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
    *, strict: bool = False,
    source_message_uuid=None, source_content_digest=None, result_observer=None, coverage_observer=None,
) -> bool:
    """Write a raw conversation turn to OSS Mem0 with inference enabled."""
    from ..extraction import parse_memory_intent
    intent = parse_memory_intent(user_content)
    if intent.kind not in {'none', 'affirmative'}:
        return False
    messages = [{'role': 'user', 'content': intent.payload or user_content[:500]}]
    if intent.kind == 'none':
        messages.append({'role': 'assistant', 'content': assistant_content[:500]})
    return add_mem0_messages(
        user_id=user_id,
        messages=messages,
        conversation_id=conversation_id,
        assistant_message_uuid=assistant_message_uuid,
        strict=strict,
        source_message_uuid=source_message_uuid, source_content_digest=source_content_digest,
        result_observer=result_observer, coverage_observer=coverage_observer,
    )


__all__ = [
    "search_mem0_episodic_memories",
    "search_mem0_episodic_memories_async",
    "search_mem0_via_oss",
    "search_mem0_via_oss_async",
    "add_mem0_messages",
    "add_mem0_raw_facts",
    "add_mem0_raw_facts_async",
    "add_mem0_memory_with_client",
]


def reconcile_known_profile_copies(*, user_id: str, provider_memory_ids: list[str], source_conversation_id=None, source_message_uuid=None, source_content_digest=None) -> None:
    """Delete only identified copies whose provider receipt still verifies owner/source."""
    if not provider_memory_ids or len(provider_memory_ids) > 50:
        raise NotImplementedError('unidentified_provider_copies')
    client = get_mem0_oss()
    if client is None or not callable(getattr(client, 'get', None)) or not callable(getattr(client, 'delete', None)):
        raise NotImplementedError('unsupported_provider_lifecycle')
    for identity in provider_memory_ids:
        if not isinstance(identity, str) or not identity or len(identity) > 256:
            raise ValueError('invalid_provider_identity')
        receipt = _run_mem0_add_call_with_timeout(client.get, identity)
        if receipt is None:
            continue  # Already absent: idempotent retry.
        if not isinstance(receipt, dict) or receipt.get('id') != identity:
            raise ValueError('provider_identity_changed')
        metadata = receipt.get('metadata')
        if not isinstance(metadata, dict) or str(metadata.get('user_id')) != str(user_id):
            raise PermissionError('provider_owner_changed')
        if source_message_uuid is not None and (
            str(metadata.get('source_message_uuid')) != str(source_message_uuid)
            or metadata.get('source_content_digest') != source_content_digest
            or metadata.get('conversation_id') != source_conversation_id
        ):
            raise PermissionError('provider_source_changed')
        result = _run_mem0_add_call_with_timeout(client.delete, identity)
        # Unknown/missing receipts are not proof of deletion.
        if not isinstance(result, dict) or result.get('message') not in {'deleted', 'Memory deleted successfully'}:
            raise NotImplementedError('unconfirmed_provider_deletion')
