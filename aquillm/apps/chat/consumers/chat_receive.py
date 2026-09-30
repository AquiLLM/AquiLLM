"""WebSocket receive handler for chat append / rate / feedback actions."""

from __future__ import annotations

import re
from base64 import b64decode
from json import dumps, loads
from time import perf_counter
from typing import Any

import structlog
from channels.db import database_sync_to_async
from django.core.exceptions import ValidationError
from django.core.files.base import ContentFile
from django.db import transaction

from apps.chat.consumers.chat_delta import send_conversation_delta
from apps.chat.consumers.chat_intent import (
    _looks_like_explicit_document_search_request,
    _looks_like_local_tool_request,
    _looks_like_retry_request,
)
from apps.chat.consumers.chat_publish import run_llm_spin
from apps.chat.consumers.chat_ws_errors import (
    send_connect_error,
    send_receive_error,
    send_receive_validation_error,
)
from apps.chat.consumers.utils import CHAT_MAX_FUNC_CALLS, CHAT_MAX_TOKENS
from apps.chat.models import ConversationFile, WSConversation
from apps.chat.services.collection_prompt_skills import validate_skill_overrides
from apps.chat.services.execution import execution_turn
from apps.chat.services.feedback import (
    apply_message_feedback_text,
    apply_message_rating,
)
from apps.chat.services.rag_config import attach_tools_when_collections_selected
from apps.chat.services.rag_intent import classify_chat_message
from apps.chat.services.rag_pipeline import run_direct_rag_turn
from apps.chat.services.rag_turn import preservation_turn
from apps.chat.services.skills_runtime import effective_base_system_for_memory_async
from aquillm.llm import ToolChoice, UserMessage
from aquillm.memory import augment_conversation_with_memory_async
from aquillm.message_adapters import ConversationConflictError

logger = structlog.stdlib.get_logger(__name__)


# Recall requests that point at *other* conversation threads (not documents).
_CHAT_HISTORY_TARGET_RE = re.compile(
    r"\b(?:past|previous|prior|earlier|old(?:er)?|other|last(?:\s+time)?)\s+"
    r"(?:chats?|conversations?|threads?|discussions?|sessions?|talks?)\b|"
    r"\b(?:chat|conversation|thread|discussion)\s+history\b",
    flags=re.IGNORECASE,
)
_CHAT_HISTORY_PHRASE_RE = re.compile(
    r"\bwhat\s+did\s+we\s+(?:discuss|talk\s+about|say|decide|cover|go\s+over)\b|"
    r"\b(?:discuss(?:ed)?|talk(?:ed)?\s+about|decide[d]?|said|mention(?:ed)?)\b[^.?!]*"
    r"\b(?:before|earlier|last\s+time|previously|in\s+(?:a|an|our|the)\s+"
    r"(?:past|previous|earlier|other)\s+(?:chat|conversation|thread))\b|"
    r"\bremind\s+me\s+what\s+we\b",
    flags=re.IGNORECASE,
)


def _looks_like_chat_history_search_request(message_content: str) -> bool:
    """True when the user asks to recall something from an earlier conversation."""
    text = message_content or ""
    if _CHAT_HISTORY_TARGET_RE.search(text):
        return True
    if re.search(
        r"\b(?:this|current|present|ongoing)\s+(?:chat|conversation|thread|discussion|session)\b",
        text,
        re.IGNORECASE,
    ):
        return False
    return bool(_CHAT_HISTORY_PHRASE_RE.search(text))


def _latest_prior_user_tool_intent(
    messages: list,
) -> tuple[list | None, ToolChoice | None]:
    for msg in reversed(messages):
        if isinstance(msg, UserMessage) and msg.tools and msg.tool_choice:
            return msg.tools, msg.tool_choice
    return None, None


def _configure_append_tools(
    *,
    message_content: str,
    all_tools: list,
    document_tools: list,
    selected_collection_ids: list | None = None,
    memory_tools: list | None = None,
    prior_user_tools: list | None = None,
    prior_user_tool_choice: ToolChoice | None = None,
) -> tuple[list, ToolChoice | None]:
    """Choose tool availability and choice strength for an appended user message."""
    if prior_user_tools and _looks_like_retry_request(message_content):
        return prior_user_tools, prior_user_tool_choice or ToolChoice(type="auto")
    if memory_tools and _looks_like_chat_history_search_request(message_content):
        return memory_tools, ToolChoice(type="any")
    if document_tools and _looks_like_explicit_document_search_request(message_content):
        return document_tools, ToolChoice(type="any")
    collection_ids = list(selected_collection_ids or [])
    if document_tools and collection_ids and attach_tools_when_collections_selected():
        intent = classify_chat_message(
            message_content or "", selected_collection_ids=collection_ids
        )
        if intent.requires_rag and not intent.requires_local_tools:
            return document_tools, ToolChoice(type="any")
    if all_tools and _looks_like_local_tool_request(message_content):
        return all_tools, ToolChoice(type="auto")
    return [], None


def restore_pending_user_tool_intent(
    convo: Any,
    *,
    all_tools: list,
    document_tools: list,
    selected_collection_ids: list | None = None,
    memory_tools: list | None = None,
) -> None:
    """Apply the append policy to a saved user turn using current permissions."""
    if not convo or not isinstance(convo[-1], UserMessage):
        return

    prior_user_tools: list | None = None
    prior_user_tool_choice: ToolChoice | None = None
    for message in convo.messages:
        if not isinstance(message, UserMessage):
            continue
        active_tools, tool_choice = _configure_append_tools(
            message_content=message.content,
            all_tools=all_tools,
            document_tools=document_tools,
            selected_collection_ids=selected_collection_ids,
            memory_tools=memory_tools,
            prior_user_tools=prior_user_tools,
            prior_user_tool_choice=prior_user_tool_choice,
        )
        if message is convo[-1]:
            message.tools = active_tools
            message.tool_choice = tool_choice
            return
        if active_tools and tool_choice:
            prior_user_tools, prior_user_tool_choice = active_tools, tool_choice


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
    logger.debug("obs.chat.receive", data_chars=len(text_data))

    @database_sync_to_async
    def _save_files(files: list[ConversationFile]) -> list[ConversationFile]:
        for file in files:
            file.save()
        return files

    @database_sync_to_async
    def _save_context_selection(
        selected_collections: list[Any], skill_overrides: dict[str, bool] | None
    ) -> dict[str, bool]:
        with transaction.atomic():
            db_convo = WSConversation.objects.select_for_update().get(
                pk=consumer.db_convo.pk
            )
            db_convo.selected_collection_ids = selected_collections
            update_fields = ["selected_collection_ids", "updated_at"]
            if skill_overrides is not None:
                db_convo.skill_overrides = skill_overrides
                update_fields.append("skill_overrides")
            db_convo.save(update_fields=update_fields)
            return dict(db_convo.skill_overrides or {})

    async def _apply_context_selection(
        selected_collections: list[Any], skill_overrides: dict[str, bool] | None
    ) -> dict[str, bool]:
        saved_overrides = await _save_context_selection(
            selected_collections, skill_overrides
        )
        consumer.db_convo.selected_collection_ids = selected_collections
        consumer.db_convo.skill_overrides = saved_overrides
        consumer.col_ref.collections = selected_collections
        consumer.skill_overrides = saved_overrides
        return saved_overrides

    async def update_selected_collections(data: dict) -> None:
        selected_collections = _validated_collection_ids(data.get("collections", []))
        overrides = (
            await database_sync_to_async(validate_skill_overrides)(
                consumer.user, data["skill_overrides"]
            )
            if "skill_overrides" in data
            else None
        )
        request_id = data.get("request_id")
        if "request_id" in data and (
            not isinstance(request_id, str) or not request_id or len(request_id) > 128
        ):
            raise ValidationError(
                "request_id must be a nonempty string of at most 128 characters"
            )
        saved_overrides = await _apply_context_selection(
            selected_collections, overrides
        )
        acknowledgement = {
            "selected_collections": selected_collections,
            "skill_overrides": saved_overrides,
        }
        if request_id is not None:
            acknowledgement["request_id"] = request_id
        await consumer.send(text_data=dumps({"context_selection": acknowledgement}))

    async def append(data: dict):
        logger.debug("obs.chat.append", collections=data.get("collections", []))

        assert consumer.convo is not None

        selected_collections = _validated_collection_ids(data.get("collections", []))
        overrides = (
            await database_sync_to_async(validate_skill_overrides)(
                consumer.user, data["skill_overrides"]
            )
            if "skill_overrides" in data
            else None
        )
        message = UserMessage.model_validate(data["message"])
        if overrides is not None:
            await _apply_context_selection(selected_collections, overrides)
        else:
            consumer.col_ref.collections = selected_collections
        consumer.convo += message
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
        prior_user_tools, prior_user_tool_choice = _latest_prior_user_tool_intent(
            consumer.convo.messages[:-1]
        )
        active_tools, tool_choice = _configure_append_tools(
            message_content=consumer.convo[-1].content,
            all_tools=consumer.tools,
            document_tools=getattr(consumer, "doc_tools", []),
            selected_collection_ids=selected_collections,
            memory_tools=getattr(consumer, "memory_tools", []),
            prior_user_tools=prior_user_tools,
            prior_user_tool_choice=prior_user_tool_choice,
        )
        consumer.convo[-1].tools = active_tools
        consumer.convo[-1].files = [(file.name, file.id) for file in files]
        consumer.convo[-1].tool_choice = tool_choice
        await consumer._save_conversation(
            create_memories=False, selected_collections=selected_collections
        )
        consumer.last_sent_sequence = len(consumer.convo) - 1
        logger.debug("obs.chat.append_completed")

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

        persisted_text = await database_sync_to_async(apply_message_feedback_text)(
            consumer.db_convo.id,
            uuid_str,
            feedback_text,
        )

        for msg in consumer.convo:
            if str(msg.message_uuid) == uuid_str:
                msg.feedback_text = persisted_text
                break

    if not consumer.dead:
        data: Any = None
        action: Any = None
        try:
            data = loads(text_data)
            action = data.pop("action", None)
            logger.debug("obs.chat.action", action=action)
            if action == "append":
                async with execution_turn(consumer):
                    await append(data)
                    augment_start = perf_counter()
                    await augment_conversation_with_memory_async(
                        consumer.convo,
                        consumer.user,
                        await effective_base_system_for_memory_async(consumer),
                        consumer.db_convo.id,
                        include_episodic=not bool(consumer.col_ref.collections),
                    )
                    logger.info(
                        "obs.chat.memory_augmented",
                        phase="receive",
                        duration_ms=(perf_counter() - augment_start) * 1000,
                    )
                    async with preservation_turn(
                        consumer, max_func_calls=CHAT_MAX_FUNC_CALLS
                    ):
                        direct_outcome = await run_direct_rag_turn(
                            consumer,
                            consumer.llm_if,
                            consumer.convo,
                            stream_func=consumer._send_stream_payload,
                        )
                        if direct_outcome == "handled":
                            await send_conversation_delta(
                                consumer,
                                consumer.convo,
                                create_memories=False,
                                close_db=True,
                            )
                        else:
                            logger.debug("obs.chat.spin_starting", phase="receive")
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
                            logger.info(
                                "obs.chat.spin_completed",
                                phase="receive",
                                duration_ms=(perf_counter() - llm_start) * 1000,
                            )
                        await consumer._save_conversation(create_memories=True)
            elif action == "select_collections":
                await update_selected_collections(data)
            elif action == "rate":
                await rate(data)
            elif action == "feedback":
                await feedback(data)
            else:
                raise ValueError(f'Invalid action "{action}"')
            logger.debug("obs.chat.action_completed", action=action)
        except ConversationConflictError as e:
            logger.warning("obs.chat.conversation_conflict", error=str(e))
            await send_connect_error(
                consumer,
                e,
                code=4409,
                message=(
                    "This chat changed in another connection. Refresh the page "
                    "and resend your message if needed."
                ),
            )
        except ValidationError as e:
            msg = e.messages[0] if getattr(e, "messages", None) else str(e)
            logger.warning("obs.chat.validation_error", error=msg)
            request_id = data.get("request_id") if isinstance(data, dict) else None
            if (
                action == "select_collections"
                and isinstance(request_id, str)
                and 0 < len(request_id) <= 128
            ):
                await consumer.send(
                    text_data=dumps(
                        {
                            "context_selection_error": {
                                "request_id": request_id,
                                "message": msg,
                            }
                        }
                    )
                )
            else:
                await send_receive_validation_error(consumer, msg)
        except Exception as e:
            logger.error(
                "obs.chat.receive_error",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )
            request_id = data.get("request_id") if isinstance(data, dict) else None
            if (
                action == "select_collections"
                and isinstance(request_id, str)
                and 0 < len(request_id) <= 128
            ):
                await consumer.send(
                    text_data=dumps(
                        {
                            "context_selection_error": {
                                "request_id": request_id,
                                "message": "Unable to save chat context. Try again.",
                            }
                        }
                    )
                )
            else:
                await send_receive_error(consumer, e)


__all__ = ["handle_chat_receive"]
