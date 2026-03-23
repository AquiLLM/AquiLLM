"""Consumers for chat app."""
from __future__ import annotations

from apps.chat.refs import ChatRef, CollectionsRef

__all__ = ["ChatConsumer", "CollectionsRef", "ChatRef"]


def __getattr__(name: str):
    if name == "ChatConsumer":
        from .chat import ChatConsumer

        return ChatConsumer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
