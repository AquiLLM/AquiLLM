"""
Memory types and data structures for user memory systems.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RetrievedEpisodicMemory:
    """A memory retrieved from episodic storage."""
    content: str
    conversation_id: Optional[int] = None


__all__ = ['RetrievedEpisodicMemory']
