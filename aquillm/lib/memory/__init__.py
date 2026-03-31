"""
User memory systems: profile facts and episodic semantic memory.

This module provides:
- Mem0 client management (cloud and OSS SDK)
- Stable fact extraction from conversation turns
- Memory formatting for system prompts
- Configuration management

For Django model interactions (UserMemoryFact, EpisodicMemory), use aquillm.memory.
"""

from .types import RetrievedEpisodicMemory
from .config import (
    EPISODIC_TOP_K,
    EPISODIC_MEMORY_MAX_CHARS,
    MEMORY_BACKEND,
    MEM0_DUAL_WRITE_LOCAL,
    MEM0_TIMEOUT_SECONDS,
    use_mem0,
)
from .mem0 import (
    add_mem0_memory_with_client,
    add_mem0_raw_facts,
    get_mem0_client,
    get_mem0_client_async,
    get_mem0_oss,
    get_mem0_oss_async,
    search_mem0_episodic_memories,
    search_mem0_episodic_memories_async,
    search_mem0_via_oss,
    search_mem0_via_oss_async,
)
from .extraction import (
    extract_stable_facts,
    heuristic_facts_from_turn,
    has_remember_intent,
    normalize_remember_fact,
)
from .formatting import format_memories_for_system

__all__ = [
    # Types
    'RetrievedEpisodicMemory',
    # Config
    'EPISODIC_TOP_K',
    'EPISODIC_MEMORY_MAX_CHARS',
    'MEMORY_BACKEND',
    'MEM0_DUAL_WRITE_LOCAL',
    'MEM0_TIMEOUT_SECONDS',
    'use_mem0',
    # Mem0 client
    "get_mem0_client",
    "get_mem0_client_async",
    "get_mem0_oss",
    "get_mem0_oss_async",
    "search_mem0_episodic_memories",
    "search_mem0_episodic_memories_async",
    "search_mem0_via_oss",
    "search_mem0_via_oss_async",
    "add_mem0_raw_facts",
    "add_mem0_memory_with_client",
    # Extraction
    'extract_stable_facts',
    'heuristic_facts_from_turn',
    'has_remember_intent',
    'normalize_remember_fact',
    # Formatting
    'format_memories_for_system',
]
