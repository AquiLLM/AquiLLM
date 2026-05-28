"""WebSocket receive handler for chat append / rate / feedback actions."""
from __future__ import annotations

import structlog
from base64 import b64decode
from json import loads
from time import perf_counter
from typing import Any

from channels.db import database_sync_to_async
from django.core.exceptions import ValidationError
from django.core.files.base import ContentFile

from aquillm.llm import ToolChoice, UserMessage
from aquillm.memory import augment_conversation_with_memory_async
from apps.chat.consumers.chat_delta import send_conversation_delta
from apps.chat.consumers.chat_ws_errors import (
    send_receive_error,
    send_receive_validation_error,
)
from apps.chat.consumers.utils import CHAT_MAX_FUNC_CALLS, CHAT_MAX_TOKENS
from apps.chat.models import ConversationFile
from apps.chat.services.feedback import apply_message_feedback_text, apply_message_rating
from lib.skills.commands import (
    find_skill_for_command,
    format_activated_skill_block,
    parse_slash_command,
)

logger = structlog.stdlib.get_logger(__name__)


async def handle_chat_receive(consumer: Any, text_data: str) -> None:
    logger.debug("ChatConsumer.receive() called with data: %s...", text_data[:100])

    @database_sync_to_async
    def _save_files(files: list[ConversationFile]) -> list[ConversationFile]:
        for file in files:
            file.save()
        return files

    async def append(data: dict) -> bool:
        """Returns True if the spin loop should be skipped (pure slash command)."""
        logger.debug("append() called with collections: %s", data.get("collections", []))

        assert consumer.convo is not None

        selected_collections = data["collections"]
        consumer.col_ref.collections = selected_collections
        consumer.convo += UserMessage.model_validate(data["message"])
        files: list[ConversationFile] = []
        if "files" in data:
            files = [
                ConversationFile(
                    file=ContentFile(b64decode(file["base64"]), name=file["filename"]),
                    conversation=consumer.db_convo,
                    name=file["filename"][-200:],
                    message_uuid=consumer.convo[-1].message_uuid,
                )
                for file in data["files"]
            ]
            await _save_files(files)
        # Keep the default tool set even when no collections are selected.
        # Restricting to non-document tools here caused an astronomy-only default.
        active_tools = consumer.tools
        consumer.convo[-1].tools = active_tools
        consumer.convo[-1].files = [(file.name, file.id) for file in files]
        consumer.convo[-1].tool_choice = ToolChoice(type="auto")
        # Slash command shortcut: /<skill_name> [args] activates a skill body
        # by appending it to the consumer's effective system prompt. Pure slash
        # commands skip the LLM call and return an acknowledgment.
        skip_spin = False
        msg_text = getattr(consumer.convo[-1], "content", "") or ""
        parsed = parse_slash_command(msg_text)
        if parsed is not None and getattr(consumer, "_skills", None):
            cmd_name, args = parsed
            skill = find_skill_for_command(consumer._skills, cmd_name)
            if skill is not None:
                if skill.name not in consumer._activated_skill_names:
                    consumer._activated_skill_blocks.append(
                        format_activated_skill_block(skill, args)
                    )
                    consumer._activated_skill_names.add(skill.name)
                    logger.info("skill_slash_activated", skill=skill.name, args=args)
                # Pure slash command: skip LLM, send synthetic ack instead
                from aquillm.llm import AssistantMessage

                ack = (
                    f"Skill `{skill.name}` activated"
                    + (f" with args `{args}`" if args else "")
                    + "."
                )
                consumer.convo += AssistantMessage(content=ack, stop_reason="end_turn")
                skip_spin = True
        await consumer._save_conversation(create_memories=False)
        consumer.last_sent_sequence = len(consumer.convo) - 1
        logger.debug("append() completed, message added")
        return skip_spin

    async def rate(data: dict):
        assert consumer.convo is not None
        uuid_str = data["uuid"]
        rating = data["rating"]

        await database_sync_to_async(apply_message_rating)(
            consumer.db_convo.id,
            uuid_str,
            rating,
        )

        for msg in consumer.convo:
            if str(msg.message_uuid) == uuid_str:
                msg.rating = int(rating)
                break

    async def feedback(data: dict):
        assert consumer.convo is not None
        uuid_str = data["uuid"]
        feedback_text = data["feedback_text"]

        await database_sync_to_async(apply_message_feedback_text)(
            consumer.db_convo.id,
            uuid_str,
            feedback_text,
        )

        for msg in consumer.convo:
            if str(msg.message_uuid) == uuid_str:
                raw = "" if feedback_text is None else str(feedback_text)
                msg.feedback_text = raw.strip() or None
                break

    if not consumer.dead:
        try:
            data = loads(text_data)
            action = data.pop("action", None)
            logger.debug("Action: %s", action)
            if action == "append":
                skip_spin = await append(data)
                if skip_spin:
                    # Pure slash command: send the synthetic ack message and stop
                    await send_conversation_delta(
                        consumer, consumer.convo, create_memories=False, close_db=True
                    )
                    logger.debug("Pure slash command — skipping LLM spin")
                else:
                    augment_start = perf_counter()
                    await augment_conversation_with_memory_async(
                        consumer.convo,
                        consumer.user,
                        consumer._effective_system_prompt(),
                        consumer.db_convo.id,
                    )
                    logger.info(
                        "Memory augmentation took %.1fms in receive()",
                        (perf_counter() - augment_start) * 1000,
                    )
                    logger.debug("About to call llm_if.spin() in receive()")
                    llm_start = perf_counter()
                    await consumer.llm_if.spin(
                        consumer.convo,
                        max_func_calls=CHAT_MAX_FUNC_CALLS,
                        max_tokens=CHAT_MAX_TOKENS,
                        send_func=lambda c: send_conversation_delta(
                            consumer, c, create_memories=False, close_db=True
                        ),
                        stream_func=consumer._send_stream_payload,
                    )
                    logger.info("LLM spin took %.1fms in receive()", (perf_counter() - llm_start) * 1000)
                    await consumer._save_conversation(create_memories=True)
            elif action == "rate":
                await rate(data)
            elif action == "feedback":
                await feedback(data)
            else:
                raise ValueError(f'Invalid action "{action}"')
            logger.debug("receive() action completed")
        except ValidationError as e:
            msg = e.messages[0] if getattr(e, "messages", None) else str(e)
            logger.warning("Validation error in receive(): %s", msg)
            await send_receive_validation_error(consumer, msg)
        except Exception as e:
            logger.error("Exception in receive(): %s", e, exc_info=True)
            await send_receive_error(consumer, e)


__all__ = ["handle_chat_receive"]
