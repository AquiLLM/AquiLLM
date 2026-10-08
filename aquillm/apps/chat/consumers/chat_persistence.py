"""Short transcript persistence and independent background publication."""

import asyncio
import structlog
from channels.db import database_sync_to_async
from apps.chat.models import WSConversation

logger = structlog.stdlib.get_logger(__name__)


async def save_chat_conversation(
    consumer, *, create_memories=False, selected_collections=None,
    skill_overrides=None, enqueue_functions=()
):
    title_request = await _persist_conversation(
        consumer, create_memories, selected_collections=selected_collections,
        skill_overrides=skill_overrides,
    )
    if title_request:
        from apps.chat.tasks.title import schedule_title

        schedule_title(*title_request)
    if create_memories:
        for enqueue in enqueue_functions:
            try:
                await asyncio.wait_for(
                    database_sync_to_async(enqueue, thread_sensitive=False)(
                        conversation_id=consumer.db_convo.id,
                        queued_updated_at=consumer.db_convo.updated_at.isoformat(),
                    ),
                    timeout=5,
                )
            except Exception as exc:
                logger.warning(
                    "obs.chat.background_enqueue_failed",
                    conversation_id=consumer.db_convo.id,
                    error_type=type(exc).__name__,
                )


@database_sync_to_async
def _persist_conversation(
    consumer, create_memories=False, *, selected_collections=None, skill_overrides=None
):
    from apps.chat.services.rag_turn_publication import conversation_publication
    from aquillm.message_adapters import save_conversation_to_db
    from lib.llm.execution_context import execution_publication

    assert consumer.db_convo is not None
    with execution_publication(), conversation_publication():
        context = {}
        if selected_collections is not None:
            context["selected_collections"] = selected_collections
        if skill_overrides is not None:
            context["skill_overrides"] = skill_overrides
        save_conversation_to_db(consumer.convo, consumer.db_convo, **context)
    if len(consumer.convo) >= 2 and not consumer.db_convo.name:
        first_user = next(
            (message.content for message in consumer.convo if message.role == "user"),
            None,
        )
        fallback = consumer.db_convo._fallback_title_from_user_message(first_user)
        from django.db.models import Q

        changed = (
            WSConversation.objects.filter(pk=consumer.db_convo.pk, name_is_manual=False)
            .filter(Q(name__isnull=True) | Q(name=""))
            .update(name=fallback)
        )
        if changed:
            consumer.db_convo.name = fallback
            return consumer.db_convo.pk, fallback
