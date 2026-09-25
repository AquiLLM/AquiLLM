"""Serialize chat work in one owned task while ASGI can deliver disconnect."""

import asyncio

import structlog

from apps.chat.services.rag_config import rag_preservation_config
from apps.chat.services.rag_turn import cancel_active_turn

logger = structlog.stdlib.get_logger(__name__)


async def stop_chat_work(consumer):
    consumer.transport_connected = False
    cancel_active_turn(consumer)
    owner = getattr(consumer, "_chat_event_owner", None)
    if owner and not owner.done():
        owner.cancel()
        # Noncooperative sync work is publication-fenced before this finite wait.
        await asyncio.wait({owner}, timeout=0.25)


async def dispatch_chat_event(consumer, message, dispatch):
    owner = getattr(consumer, "_chat_event_owner", None)
    if owner is None and not rag_preservation_config().active:
        return await dispatch(message)
    if message["type"] == "websocket.disconnect":
        await stop_chat_work(consumer)
        return await dispatch(message)
    if owner is None:
        queue = consumer._chat_event_queue = asyncio.Queue(maxsize=32)

        async def run():
            try:
                while True:
                    event = await queue.get()
                    try:
                        await dispatch(event)
                    finally:
                        queue.task_done()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                consumer.dead = True
                logger.error(
                    "obs.chat.event_owner_failed", error_type=type(exc).__name__
                )
                await consumer.close(code=1011)
            finally:
                while not queue.empty():
                    queue.get_nowait()
                    queue.task_done()

        owner = consumer._chat_event_owner = asyncio.create_task(run())
        # Retrieve terminal errors even if ASGI disconnect finishes first.
        owner.add_done_callback(
            lambda task: None if task.cancelled() else task.exception()
        )
    if owner.done():
        return
    try:
        consumer._chat_event_queue.put_nowait(dict(message))
    except asyncio.QueueFull:
        cancel_active_turn(consumer)
        owner.cancel()
        consumer.dead = True
        await consumer.close(code=1013)
