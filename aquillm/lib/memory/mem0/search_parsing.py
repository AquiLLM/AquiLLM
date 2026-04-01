"""Parsing helpers for Mem0 search responses."""

from __future__ import annotations

from typing import Any, Optional

from ..types import RetrievedEpisodicMemory


def response_to_raw_items(response: Any) -> Any:
    """Normalize Mem0 response payload to a list-compatible raw items payload."""
    if isinstance(response, dict):
        return response.get("results") or response.get("memories") or response.get("data") or response.get("items") or []
    if isinstance(response, list):
        return response
    return []


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


def parse_mem0_search_items(
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


__all__ = ["parse_mem0_search_items", "response_to_raw_items"]
