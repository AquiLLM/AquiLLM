"""
Mem0 memory backend integration.
"""

from .client import get_mem0_client, get_mem0_client_async, get_mem0_oss, get_mem0_oss_async
from .operations import (
    add_mem0_memory_with_client,
    add_mem0_raw_facts,
    add_mem0_raw_facts_async,
    search_mem0_episodic_memories,
    search_mem0_episodic_memories_async,
    search_mem0_via_oss,
    search_mem0_via_oss_async,
)

__all__ = [
    "get_mem0_client",
    "get_mem0_client_async",
    "get_mem0_oss",
    "get_mem0_oss_async",
    "search_mem0_episodic_memories",
    "search_mem0_episodic_memories_async",
    "search_mem0_via_oss",
    "search_mem0_via_oss_async",
    "add_mem0_raw_facts",
    "add_mem0_raw_facts_async",
    "add_mem0_memory_with_client",
]
