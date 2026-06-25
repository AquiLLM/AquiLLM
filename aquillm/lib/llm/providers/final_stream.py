"""Final-answer streaming helpers for provider orchestration."""
from __future__ import annotations

from os import getenv
from typing import Any, Awaitable, Callable


def final_answer_streaming_enabled() -> bool:
    """Default to streaming only after an answer has passed server-side cleanup."""
    return getenv("LLM_STREAM_FINAL_ANSWER_ONLY", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


async def send_final_answer_stream(
    stream_func: Callable[[dict], Awaitable[Any]],
    *,
    message_uuid: str,
    content: str,
    stop_reason: str | None,
    usage: int,
) -> None:
    """Publish the validated assistant answer as the sole visible stream event."""
    if not content.strip():
        return
    await stream_func(
        {
            "message_uuid": message_uuid,
            "role": "assistant",
            "content": content,
            "done": True,
            "stop_reason": stop_reason,
            "usage": usage,
        }
    )


__all__ = [
    "final_answer_streaming_enabled",
    "send_final_answer_stream",
]
