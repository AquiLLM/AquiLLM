"""Mutable reference holders for chat WebSocket tooling (closures over consumer state)."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from apps.chat.consumers.chat import ChatConsumer


class CollectionsRef:
    """Reference holder for collections, allowing mutation inside closures."""

    def __init__(self, collections: list[int]):
        self.collections = collections


class ChatRef:
    """Reference holder for ChatConsumer, allowing mutation inside closures."""

    def __init__(self, chat: ChatConsumer):
        self.chat = chat


__all__ = ["CollectionsRef", "ChatRef"]
