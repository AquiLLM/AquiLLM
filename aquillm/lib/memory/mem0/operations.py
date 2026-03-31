"""
Mem0 search and write operations.
"""

from __future__ import annotations

import asyncio
import structlog
from os import getenv
from typing import Any, Optional

from ..config import MEM0_TIMEOUT_SECONDS
from ..types import RetrievedEpisodicMemory
from .client import get_mem0_client, get_mem0_client_async, get_mem0_oss, get_mem0_oss_async

logger = structlog.stdlib.get_logger(__name__)


def _extract_mem0_content(item: Any) -> str:
    """Extract text content from a Mem0 result item."""
    if isinstance(item, str):
        return item.strip()
    if not isinstance(item, dict):
        return str(item).strip()
    for key in ("memory", "text", "content", "value"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return str(item).strip()


def _extract_mem0_conversation_id(item: Any) -> Optional[int]:
    """Extract conversation ID from Mem0 result metadata."""
    if not isinstance(item, dict):
        return None
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    raw_value = metadata.get("conversation_id")
    if raw_value is None:
        return None
    try:
        return int(raw_value)
    except Exception:
        return None


def _parse_mem0_search_items(
    raw_items: Any, top_k: int, exclude_conversation_id: Optional[int]
) -> list[RetrievedEpisodicMemory]:
    """Parse Mem0 search results into RetrievedEpisodicMemory objects."""
    parsed: list[RetrievedEpisodicMemory] = []
    if not isinstance(raw_items, list):
        return parsed
    for item in raw_items:
        content = _extract_mem0_content(item)
        if not content:
            continue
        conv_id = _extract_mem0_conversation_id(item)
        if exclude_conversation_id is not None and conv_id == exclude_conversation_id:
            continue
        parsed.append(RetrievedEpisodicMemory(content=content, conversation_id=conv_id))
    return parsed[:top_k]


async def _await_mem0_search(mem0: Any, args: tuple, kwargs: dict) -> Any:
    return await mem0.search(*args, **kwargs)  # type: ignore[attr-defined]


def search_mem0_via_oss(
    user_id: str, query: str, top_k: int, exclude_conversation_id: Optional[int]
) -> list[RetrievedEpisodicMemory]:
    """
    Query local/self-hosted Mem0 via OSS SDK first.
    SDK signatures vary across versions, so we try several call shapes.
    """
    mem0 = get_mem0_oss()
    if mem0 is None:
        return []

    call_attempts = [
        ((), {"query": query, "user_id": user_id, "limit": top_k}),
        ((), {"query": query, "user_id": user_id, "top_k": top_k}),
        ((), {"query": query, "user_id": user_id}),
        ((query,), {"user_id": user_id, "limit": top_k}),
        ((query,), {"user_id": user_id}),
    ]

    response = None
    for args, kwargs in call_attempts:
        try:
            response = mem0.search(*args, **kwargs)  # type: ignore[attr-defined]
            break
        except TypeError:
            continue
        except Exception as exc:
            logger.warning("Mem0 OSS search failed; falling back. Error: %s", exc)
            return []

    if response is None:
        logger.warning("Mem0 OSS search call signature not supported; falling back.")
        return []

    if isinstance(response, dict):
        raw_items = (
            response.get("results")
            or response.get("memories")
            or response.get("data")
            or response.get("items")
            or []
        )
    elif isinstance(response, list):
        raw_items = response
    else:
        raw_items = []

    return _parse_mem0_search_items(
        raw_items=raw_items,
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

    call_attempts = [
        ((), {"query": query, "user_id": user_id, "limit": top_k}),
        ((), {"query": query, "user_id": user_id, "top_k": top_k}),
        ((), {"query": query, "user_id": user_id}),
        ((query,), {"user_id": user_id, "limit": top_k}),
        ((query,), {"user_id": user_id}),
    ]

    response = None
    for args, kwargs in call_attempts:
        try:
            response = await asyncio.wait_for(
                _await_mem0_search(mem0, args, kwargs),
                timeout=float(MEM0_TIMEOUT_SECONDS),
            )
            break
        except TypeError:
            continue
        except asyncio.TimeoutError:
            logger.warning("Mem0 OSS async search timed out after %ss", MEM0_TIMEOUT_SECONDS)
            return []
        except Exception as exc:
            logger.warning("Mem0 OSS async search failed; falling back. Error: %s", exc)
            return []

    if response is None:
        logger.warning("Mem0 OSS async search call signature not supported; falling back.")
        return []

    if isinstance(response, dict):
        raw_items = (
            response.get("results")
            or response.get("memories")
            or response.get("data")
            or response.get("items")
            or []
        )
    elif isinstance(response, list):
        raw_items = response
    else:
        raw_items = []

    return _parse_mem0_search_items(
        raw_items=raw_items,
        top_k=top_k,
        exclude_conversation_id=exclude_conversation_id,
    )


def search_mem0_episodic_memories(
    user_id: str, query: str, top_k: int, exclude_conversation_id: Optional[int]
) -> list[RetrievedEpisodicMemory]:
    """Search Mem0 for episodic memories, trying OSS SDK first then cloud client."""
    oss_results = search_mem0_via_oss(
        user_id=user_id, query=query, top_k=top_k, exclude_conversation_id=exclude_conversation_id
    )
    if oss_results:
        return oss_results

    if not getenv("MEM0_API_KEY"):
        return []
    client = get_mem0_client()
    if client is None:
        return []
    try:
        response = client.search(  # type: ignore[attr-defined]
            query=query,
            user_id=user_id,
            limit=top_k,
        )
        if isinstance(response, dict):
            raw_items = response.get("results") or response.get("memories") or response.get("data") or []
        elif isinstance(response, list):
            raw_items = response
        else:
            raw_items = []

        return _parse_mem0_search_items(
            raw_items=raw_items,
            top_k=top_k,
            exclude_conversation_id=exclude_conversation_id,
        )
    except Exception as exc:
        logger.warning("Mem0 search failed; falling back to local memory. Error: %s", exc)
        return []


async def search_mem0_episodic_memories_async(
    user_id: str, query: str, top_k: int, exclude_conversation_id: Optional[int]
) -> list[RetrievedEpisodicMemory]:
    """Search Mem0 for episodic memories using async OSS SDK first, then async cloud client."""
    oss_results = await search_mem0_via_oss_async(
        user_id=user_id, query=query, top_k=top_k, exclude_conversation_id=exclude_conversation_id
    )
    if oss_results:
        return oss_results

    if not getenv("MEM0_API_KEY"):
        return []
    client = await get_mem0_client_async()
    if client is None:
        return []
    try:
        response = await asyncio.wait_for(
            client.search(  # type: ignore[attr-defined]
                query=query,
                user_id=user_id,
                limit=top_k,
            ),
            timeout=float(MEM0_TIMEOUT_SECONDS),
        )
        if isinstance(response, dict):
            raw_items = response.get("results") or response.get("memories") or response.get("data") or []
        elif isinstance(response, list):
            raw_items = response
        else:
            raw_items = []

        return _parse_mem0_search_items(
            raw_items=raw_items,
            top_k=top_k,
            exclude_conversation_id=exclude_conversation_id,
        )
    except asyncio.TimeoutError:
        logger.warning("Mem0 cloud async search timed out after %ss", MEM0_TIMEOUT_SECONDS)
        return []
    except Exception as exc:
        logger.warning("Mem0 async search failed; falling back to local memory. Error: %s", exc)
        return []


def add_mem0_raw_facts(
    user_id: str,
    facts: list[str],
    conversation_id: int,
    assistant_message_uuid: str,
) -> bool:
    """Write already-extracted facts into Mem0 with infer=False via OSS SDK."""
    mem0 = get_mem0_oss()
    if mem0 is None:
        return False

    wrote_any = False
    for fact in facts:
        try:
            result = mem0.add(  # type: ignore[attr-defined]
                fact,
                user_id=user_id,
                metadata={
                    "conversation_id": conversation_id,
                    "assistant_message_uuid": assistant_message_uuid,
                    "source": "aquillm",
                    "memory_type": "episodic",
                },
                infer=False,
            )
            if isinstance(result, dict):
                events = result.get("results") or []
                if any(isinstance(x, dict) and x.get("event") == "ADD" for x in events):
                    wrote_any = True
            else:
                wrote_any = True
        except Exception as exc:
            logger.warning("Mem0 raw fact add failed for fact=%r: %s", fact, exc)
    return wrote_any


def add_mem0_memory_with_client(
    user_id: str,
    user_content: str,
    assistant_content: str,
    conversation_id: int,
    assistant_message_uuid: str,
) -> None:
    """Add memory using Mem0 cloud client as fallback."""
    if not getenv("MEM0_API_KEY"):
        return
    client = get_mem0_client()
    if client is None:
        return
    try:
        client.add(  # type: ignore[attr-defined]
            messages=[
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ],
            user_id=user_id,
            metadata={
                "conversation_id": conversation_id,
                "assistant_message_uuid": assistant_message_uuid,
                "source": "aquillm",
                "memory_type": "episodic",
            },
        )
    except Exception as exc:
        logger.warning("Mem0 add failed; continuing with local memory. Error: %s", exc)


__all__ = [
    "search_mem0_episodic_memories",
    "search_mem0_episodic_memories_async",
    "search_mem0_via_oss",
    "search_mem0_via_oss_async",
    "add_mem0_raw_facts",
    "add_mem0_memory_with_client",
]
