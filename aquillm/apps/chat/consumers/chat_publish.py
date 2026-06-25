"""WebSocket publish policy while the LLM tool loop (spin) is in flight."""
from __future__ import annotations

import structlog
from json import dumps
from typing import Any, Awaitable, Callable

from aquillm.llm import Conversation
from aquillm.message_adapters import pydantic_message_to_frontend_dict

logger = structlog.stdlib.get_logger(__name__)


def begin_spin_publish(consumer: Any) -> None:
    """Mark the start of a multi-step tool loop for UI publishing."""
    consumer._spin_active = True
    consumer._spin_epoch = len(consumer.convo) if consumer.convo is not None else 0


async def end_spin_publish(consumer: Any, convo: Conversation) -> None:
    """
    After spin completes, push display-ready assistant content for messages
    that were held back (empty bubble / spinner only) during the loop.
    """
    consumer._spin_active = False
    if consumer.convo is None:
        consumer.convo = convo
    epoch = int(getattr(consumer, "_spin_epoch", 0))
    if epoch >= len(convo):
        return
    refresh_messages = [
        pydantic_message_to_frontend_dict(msg) for msg in convo.messages[epoch:]
    ]
    if not refresh_messages:
        return
    await consumer.send(text_data=dumps({"delta": {"messages": refresh_messages}}))
    logger.debug(
        "spin_finalize_publish messages=%d epoch=%d",
        len(refresh_messages),
        epoch,
    )


async def run_llm_spin(
    consumer: Any,
    llm_if: Any,
    convo: Conversation,
    *,
    max_func_calls: int,
    max_tokens: int,
    send_func: Callable[..., Awaitable[Any]],
    stream_func: Callable[[dict], Awaitable[Any]] | None = None,
) -> None:
    """
    Run the tool loop.

    Live streaming stays on; ``visibility`` filters interim tokens. Deltas use empty
    assistant prose until spin ends, then ``end_spin_publish`` refreshes display text.
    """
    begin_spin_publish(consumer)
    try:
        await llm_if.spin(
            convo,
            max_func_calls=max_func_calls,
            max_tokens=max_tokens,
            send_func=send_func,
            stream_func=stream_func,
        )
    finally:
        await end_spin_publish(consumer, convo)


__all__ = [
    "begin_spin_publish",
    "end_spin_publish",
    "run_llm_spin",
]
