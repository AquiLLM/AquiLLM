"""WebSocket receive handler for chat append / rate / feedback actions."""
from __future__ import annotations

import structlog
import re
from base64 import b64decode
from json import loads
from time import perf_counter
from typing import Any, Optional

from channels.db import database_sync_to_async
from django.core.exceptions import ValidationError
from django.core.files.base import ContentFile

from aquillm.llm import ToolChoice, UserMessage
from aquillm.memory import augment_conversation_with_memory_async
from apps.chat.consumers.chat_delta import send_conversation_delta
from apps.chat.consumers.chat_publish import run_llm_spin
from apps.chat.consumers.chat_ws_errors import (
    send_receive_error,
    send_receive_validation_error,
)
from apps.chat.consumers.utils import CHAT_MAX_FUNC_CALLS, CHAT_MAX_TOKENS
from apps.chat.models import ConversationFile
from apps.chat.services.feedback import apply_message_feedback_text, apply_message_rating
from apps.chat.services.skills_runtime import effective_base_system_for_memory

logger = structlog.stdlib.get_logger(__name__)


_DOCUMENT_TARGET_RE = re.compile(
    r"\b(documents?|docs?|papers?|files?|selected collections?|sources?)\b",
    flags=re.IGNORECASE,
)
_DOCUMENT_SEARCH_ACTION_RE = re.compile(
    r"\b(search|check|find|scan|read|retrieve|query|consult)\b|"
    r"\blook\s+(?:at|in|through|up)\b",
    flags=re.IGNORECASE,
)
_DOCUMENT_FIGURE_TARGET_RE = re.compile(
    r"\b(figures?|figs?\.?|images?|visuals?|plots?|graphs?|charts?|diagrams?)\b",
    flags=re.IGNORECASE,
)
_DOCUMENT_FIGURE_ACTION_RE = re.compile(
    r"\b(show|display|render|include|explain|find|get|pull|open)\b",
    flags=re.IGNORECASE,
)
_LOCAL_TOOL_ACTION_RE = re.compile(
    r"\b("
    r"sky\s+subtraction|subtract\s+the\s+sky|flat[-\s]?field(?:ing)?|"
    r"point\s+source(?:s)?|detect\s+source(?:s)?|fits|uploaded\s+files?|"
    r"use\s+(?:the\s+)?tool|run\s+(?:the\s+)?tool"
    r")\b",
    flags=re.IGNORECASE,
)


def _looks_like_explicit_document_search_request(message_content: str) -> bool:
    """True when the user explicitly asks the assistant to retrieve from documents."""
    text = message_content or ""
    figure_request = _DOCUMENT_FIGURE_TARGET_RE.search(text) and _DOCUMENT_FIGURE_ACTION_RE.search(text)
    if figure_request:
        return True
    return bool(_DOCUMENT_TARGET_RE.search(text) and _DOCUMENT_SEARCH_ACTION_RE.search(text))


def _looks_like_local_tool_request(message_content: str) -> bool:
    """True for app-local non-document tools such as FITS processing."""
    return bool(_LOCAL_TOOL_ACTION_RE.search(message_content or ""))


def _configure_append_tools(
    *,
    message_content: str,
    all_tools: list,
    document_tools: list,
) -> tuple[list, Optional[ToolChoice]]:
    """Choose tool availability and choice strength for an appended user message."""
    if document_tools and _looks_like_explicit_document_search_request(message_content):
        return document_tools, ToolChoice(type="any")
    if all_tools and _looks_like_local_tool_request(message_content):
        return all_tools, ToolChoice(type="auto")
    return [], None


def _validated_collection_ids(raw_collections: Any) -> list[Any]:
    if not isinstance(raw_collections, list):
        raise ValidationError("collections must be a list")

    collection_ids = []
    for collection_id in raw_collections:
        if isinstance(collection_id, bool) or not isinstance(collection_id, (int, str)):
            raise ValidationError("collection ids must be strings or integers")
        collection_ids.append(collection_id)
    return collection_ids


async def handle_chat_receive(consumer: Any, text_data: str) -> None:
    logger.debug("ChatConsumer.receive() called with data: %s...", text_data[:100])

    @database_sync_to_async
    def _save_files(files: list[ConversationFile]) -> list[ConversationFile]:
        for file in files:
            file.save()
        return files

    @database_sync_to_async
    def _save_selected_collections(selected_collections: list[Any]) -> None:
        consumer.db_convo.selected_collection_ids = selected_collections
        consumer.db_convo.save(update_fields=["selected_collection_ids", "updated_at"])

    async def update_selected_collections(data: dict) -> None:
        selected_collections = _validated_collection_ids(data.get("collections", []))
        consumer.col_ref.collections = selected_collections
        await _save_selected_collections(selected_collections)

    async def append(data: dict):
        logger.debug("append() called with collections: %s", data.get("collections", []))

        assert consumer.convo is not None

        selected_collections = _validated_collection_ids(data.get("collections", []))
        consumer.col_ref.collections = selected_collections
        await _save_selected_collections(selected_collections)
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
        active_tools, tool_choice = _configure_append_tools(
            message_content=consumer.convo[-1].content,
            all_tools=consumer.tools,
            document_tools=getattr(consumer, "doc_tools", []),
        )
        consumer.convo[-1].tools = active_tools
        consumer.convo[-1].files = [(file.name, file.id) for file in files]
        consumer.convo[-1].tool_choice = tool_choice
        await consumer._save_conversation(create_memories=False)
        consumer.last_sent_sequence = len(consumer.convo) - 1
        logger.debug("append() completed, message added")

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
                await append(data)
                augment_start = perf_counter()
                await augment_conversation_with_memory_async(
                    consumer.convo,
                    consumer.user,
                    effective_base_system_for_memory(consumer),
                    consumer.db_convo.id,
                )
                logger.info(
                    "Memory augmentation took %.1fms in receive()",
                    (perf_counter() - augment_start) * 1000,
                )
                logger.debug("About to call llm_if.spin() in receive()")
                llm_start = perf_counter()
                await run_llm_spin(
                    consumer,
                    consumer.llm_if,
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
            elif action == "select_collections":
                await update_selected_collections(data)
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
