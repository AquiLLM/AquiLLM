"""Build and send WebSocket conversation deltas after LLM turns."""
from __future__ import annotations

import structlog
from json import dumps
from time import perf_counter
from typing import Any

from channels.db import aclose_old_connections

from aquillm.llm import Conversation
from aquillm.message_adapters import pydantic_message_to_frontend_dict

logger = structlog.stdlib.get_logger(__name__)


async def send_conversation_delta(
    consumer: Any,
    convo: Conversation,
    *,
    create_memories: bool = False,
    close_db: bool = False,
) -> None:
    if close_db:
        await aclose_old_connections()
    logger.debug("send_func called")
    consumer.convo = convo
    save_start = perf_counter()
    await consumer._save_conversation(create_memories=create_memories)
    new_messages = convo.messages[consumer.last_sent_sequence + 1 :]
    if not new_messages:
        logger.debug("send_func skipped; no new messages to send")
        return
    usage = next(
        (
            msg.usage
            for msg in reversed(new_messages)
            if getattr(msg, "role", None) == "assistant" and getattr(msg, "usage", 0)
        ),
        None,
    )
    delta: dict[str, Any] = {
        "messages": [pydantic_message_to_frontend_dict(msg) for msg in new_messages],
    }
    if usage is not None:
        delta["usage"] = usage
    await consumer.send(text_data=dumps({"delta": delta}))
    consumer.last_sent_sequence = len(convo) - 1
    logger.info(
        "Chat send_func persisted+sent delta in %.1fms (messages=%d)",
        (perf_counter() - save_start) * 1000,
        len(new_messages),
    )
    logger.debug("send_func completed")


__all__ = ["send_conversation_delta"]
