"""WebSocket delivery boundary for chat consumers."""

from __future__ import annotations

from typing import Any


def is_client_disconnect(exc: BaseException) -> bool:
    """Recognize transport-closure signals without hiding application errors."""
    if isinstance(exc, (BrokenPipeError, ConnectionResetError)):
        return True
    exc_type = type(exc)
    if exc_type.__name__ in {"ClientDisconnected", "StopConsumer"} and (
        exc_type.__module__.startswith("channels")
        or exc_type.__module__.startswith("daphne")
    ):
        return True
    if not isinstance(exc, RuntimeError):
        return False
    message = str(exc).casefold()
    return "websocket.send" in message and (
        "websocket.close" in message or "after response completed" in message
    )


async def best_effort_send(consumer: Any, *, text_data: str) -> bool:
    """Send one payload and report whether it reached the transport."""
    if not bool(getattr(consumer, "transport_connected", True)):
        return False
    try:
        await consumer.send(text_data=text_data)
    except BaseException as exc:
        if not is_client_disconnect(exc):
            raise
        consumer.transport_connected = False
        return False
    return True


__all__ = ["best_effort_send", "is_client_disconnect"]
