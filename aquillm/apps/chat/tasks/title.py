"""Optional title refinement never delays answer publication or DB work."""

import asyncio
from os import getenv

import structlog
from asgiref.sync import async_to_sync
from celery import shared_task
from django.apps import apps

logger = structlog.stdlib.get_logger(__name__)
_enqueues = set()


def schedule_title(conversation_id, fallback):
    # Broker outages must not accumulate unbounded detached work in an ASGI worker.
    if len(_enqueues) >= 16:
        return

    async def enqueue():
        try:
            await asyncio.to_thread(
                refine_conversation_title.apply_async,
                args=[conversation_id, fallback],
                retry=False,
            )
        except Exception:
            logger.warning(
                "obs.chat.title_enqueue_failed", conversation_id=conversation_id
            )

    task = asyncio.create_task(enqueue())
    _enqueues.add(task)
    task.add_done_callback(_enqueues.discard)


@shared_task(serializer="json", soft_time_limit=15, time_limit=20)
def refine_conversation_title(conversation_id, fallback):
    from apps.chat.models import WSConversation

    row = WSConversation.objects.filter(pk=conversation_id, name=fallback).first()
    if row is None:
        return
    prompt = (
        row.db_messages.filter(role="user")
        .order_by("sequence_number")
        .values_list("content", flat=True)
        .first()
    )
    if not prompt:
        return
    llm = apps.get_app_config("aquillm").llm_interface

    async def generate():
        timeout = min(10.0, max(0.1, float(getenv("CHAT_TITLE_TIMEOUT_SECONDS", "8"))))
        return await asyncio.wait_for(
            llm.get_message(
                **(
                    llm.base_args
                    | {
                        "max_tokens": 30,
                        "thinking_budget": 0,
                        "system": "Return only a brief 3 to 10 word title describing the user's question.",
                        "messages": [{"role": "user", "content": prompt}],
                    }
                )
            ),
            timeout,
        )

    try:
        response = async_to_sync(generate)()
        title = row._clean_generated_title(response.text)
        if not row._is_generic_title(title):
            # Compare-and-set preserves newer/manual titles; metadata refinement
            # deliberately leaves activity/transcript identity unchanged.
            WSConversation.objects.filter(pk=row.pk, name=fallback).update(name=title)
    except Exception:
        logger.warning("obs.chat.auto_title_failed", conversation_id=conversation_id)
