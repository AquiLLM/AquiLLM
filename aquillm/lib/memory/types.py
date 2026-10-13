"""
Memory types and data structures for user memory systems.
"""

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass(frozen=True)
class MemoryIntent:
    """Conservative persistence intent and exact user-authored payload coordinates."""
    kind: Literal['affirmative', 'negated', 'quoted', 'hypothetical', 'ambiguous', 'none']
    payload: Optional[str] = None
    start: Optional[int] = None
    end: Optional[int] = None


@dataclass
class RetrievedEpisodicMemory:
    """A memory retrieved from episodic storage."""
    content: str
    conversation_id: Optional[int] = None
    provider_memory_id: Optional[str] = None
    source_message_uuid: Optional[str] = None
    source_content_digest: Optional[str] = None


__all__ = ['RetrievedEpisodicMemory', 'MemoryIntent']
