"""WebSocket consumer for chat functionality."""
from __future__ import annotations

import logging
import sys
from base64 import b64decode
from json import dumps, loads
from os import getenv
from time import perf_counter
from typing import Any, Optional

from anthropic._exceptions import OverloadedError
from asgiref.sync import async_to_sync
from channels.db import aclose_old_connections, database_sync_to_async
from channels.generic.websocket import AsyncWebsocketConsumer
from django.apps import apps
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
from django.core.files.base import ContentFile

from aquillm.llm import (
    Conversation,
    LLMInterface,
    LLMTool,
    ToolChoice,
    UserMessage,
    message_to_user,
)
from aquillm.memory import augment_conversation_with_memory
from aquillm.message_adapters import (
    load_conversation_from_db,
    pydantic_message_to_frontend_dict,
    save_conversation_to_db,
)
from aquillm.settings import DEBUG
from aquillm.tasks import create_conversation_memories_task
from apps.chat.models import ConversationFile, WSConversation
from apps.chat.refs import ChatRef, CollectionsRef
from apps.chat.services.feedback import apply_message_feedback_text, apply_message_rating
from apps.chat.consumers.utils import CHAT_MAX_FUNC_CALLS, CHAT_MAX_TOKENS
from apps.chat.services.tool_wiring import build_astronomy_tools, build_document_tools
from lib.tools.debug.weather import get_debug_weather_tool

logger = logging.getLogger(__name__)


class ChatConsumer(AsyncWebsocketConsumer):
    llm_if: LLMInterface = apps.get_app_config("aquillm").llm_interface
    db_convo: Optional[WSConversation] = None
    convo: Optional[Conversation] = None
    tools: list[LLMTool] = []
    user: Optional[User] = None

    dead: bool = False

    col_ref = CollectionsRef([])
    last_sent_sequence: int = -1

    async def _send_stream_payload(self, payload: dict) -> None:
        await self.send(text_data=dumps({"stream": payload}))

    async def _send_conversation_delta(
        self,
        convo: Conversation,
        *,
        create_memories: bool = False,
        close_db: bool = False,
    ) -> None:
        if close_db:
            await aclose_old_connections()
        logger.debug("send_func called")
        self.convo = convo
        save_start = perf_counter()
        await self.__save(create_memories=create_memories)
        new_messages = convo.messages[self.last_sent_sequence + 1 :]
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
        await self.send(text_data=dumps({"delta": delta}))
        self.last_sent_sequence = len(convo) - 1
        logger.info(
            "Chat send_func persisted+sent delta in %.1fms (messages=%d)",
            (perf_counter() - save_start) * 1000,
            len(new_messages),
        )
        logger.debug("send_func completed")

    @database_sync_to_async
    def __save(self, create_memories: bool = False):
        assert self.db_convo is not None
        save_conversation_to_db(self.convo, self.db_convo)
        if create_memories:
            try:
                create_conversation_memories_task.delay(self.db_convo.id)
            except Exception as exc:
                logger.warning(
                    "Failed to queue memory extraction task for convo %s: %s",
                    self.db_convo.id,
                    exc,
                )
        if len(self.convo) >= 2 and not self.db_convo.name:
            self.db_convo.set_name()

    @database_sync_to_async
    def __get_convo(self, convo_id: int, user: User):
        convo = WSConversation.objects.filter(id=convo_id).first()
        if convo:
            if convo.owner == user:
                return convo
            return None
        return convo

    @database_sync_to_async
    def __get_all_user_collections(self):
        from apps.collections.models import CollectionPermission

        self.col_ref.collections = [
            col_perm.collection.id
            for col_perm in CollectionPermission.objects.filter(user=self.user)
        ]

    async def connect(self):
        logger.debug("ChatConsumer.connect() called")

        await self.accept()
        logger.debug("WebSocket accepted")
        self.user = self.scope["user"]
        assert self.user is not None
        logger.debug("User: %s", self.user)
        await self.__get_all_user_collections()
        logger.debug("Collections loaded: %s", self.col_ref.collections)
        self.doc_tools = build_document_tools(self.user, self.col_ref, ChatRef(self))
        self.tools = self.doc_tools + build_astronomy_tools(self)
        if getenv("LLM_CHOICE") == "GEMMA3":
            self.tools.append(message_to_user)
        if DEBUG:
            self.tools.append(get_debug_weather_tool())
        convo_id = self.scope["url_route"]["kwargs"]["convo_id"]
        logger.debug("Convo ID: %s", convo_id)
        self.db_convo = await self.__get_convo(convo_id, self.user)
        if self.db_convo is None:
            logger.error("Invalid conversation ID: %s", convo_id)
            self.dead = True
            await self.send('{"exception": "Invalid chat_id"}')
            return
        try:
            self.convo = await database_sync_to_async(load_conversation_from_db)(self.db_convo)
            self.last_sent_sequence = len(self.convo) - 1
            await self.send(
                text_data=dumps(
                    {
                        "conversation": {
                            "system": self.db_convo.system_prompt,
                            "messages": [
                                pydantic_message_to_frontend_dict(msg) for msg in self.convo
                            ],
                        }
                    }
                )
            )
            augment_start = perf_counter()
            await database_sync_to_async(augment_conversation_with_memory)(
                self.convo, self.user, self.db_convo.system_prompt, self.db_convo.id
            )
            logger.info(
                "Memory augmentation took %.1fms in connect()",
                (perf_counter() - augment_start) * 1000,
            )
            self.convo.rebind_tools(self.tools)
            logger.debug("About to call llm_if.spin() in connect()")
            before_spin_len = len(self.convo)
            llm_start = perf_counter()
            await self.llm_if.spin(
                self.convo,
                max_func_calls=CHAT_MAX_FUNC_CALLS,
                max_tokens=CHAT_MAX_TOKENS,
                send_func=lambda c: self._send_conversation_delta(c, create_memories=False, close_db=False),
                stream_func=self._send_stream_payload,
            )
            logger.info("LLM spin took %.1fms in connect()", (perf_counter() - llm_start) * 1000)
            await self.__save(create_memories=len(self.convo) > before_spin_len)
            logger.debug("llm_if.spin() completed in connect()")
            return
        except OverloadedError as e:
            logger.error("LLM overloaded: %s", e)
            self.dead = True
            await self.send('{"exception": "LLM provider is currently overloaded. Try again later."}')
            return
        except Exception as e:
            logger.error("Exception in connect(): %s", e, exc_info=True)
            if DEBUG:
                from django.views.debug import ExceptionReporter

                reporter = ExceptionReporter(None, *sys.exc_info())
                debug_html = reporter.get_traceback_html()
                await self.send(text_data=dumps({"exception": str(e), "debug_html": debug_html}))
            else:
                await self.send(
                    text_data='{"exception": "A server error has occurred. Try reloading the page"}'
                )
            return

    async def receive(self, text_data):
        logger.debug("ChatConsumer.receive() called with data: %s...", text_data[:100])

        @database_sync_to_async
        def _save_files(files: list[ConversationFile]) -> list[ConversationFile]:
            for file in files:
                file.save()
            return files

        async def append(data: dict):
            logger.debug("append() called with collections: %s", data.get("collections", []))

            assert self.convo is not None

            selected_collections = data["collections"]
            self.col_ref.collections = selected_collections
            self.convo += UserMessage.model_validate(data["message"])
            files: list[ConversationFile] = []
            if "files" in data:
                files = [
                    ConversationFile(
                        file=ContentFile(b64decode(file["base64"]), name=file["filename"]),
                        conversation=self.db_convo,
                        name=file["filename"][-200:],
                        message_uuid=self.convo[-1].message_uuid,
                    )
                    for file in data["files"]
                ]
                await _save_files(files)
            active_tools = (
                self.tools
                if selected_collections
                else [tool for tool in self.tools if tool not in self.doc_tools]
            )
            self.convo[-1].tools = active_tools
            self.convo[-1].files = [(file.name, file.id) for file in files]
            self.convo[-1].tool_choice = ToolChoice(type="auto")
            await self.__save(create_memories=False)
            self.last_sent_sequence = len(self.convo) - 1
            logger.debug("append() completed, message added")

        async def rate(data: dict):
            assert self.convo is not None
            uuid_str = data["uuid"]
            rating = data["rating"]

            await database_sync_to_async(apply_message_rating)(
                self.db_convo.id,
                uuid_str,
                rating,
            )

            for msg in self.convo:
                if str(msg.message_uuid) == uuid_str:
                    msg.rating = int(rating)
                    break

        async def feedback(data: dict):
            assert self.convo is not None
            uuid_str = data["uuid"]
            feedback_text = data["feedback_text"]

            await database_sync_to_async(apply_message_feedback_text)(
                self.db_convo.id,
                uuid_str,
                feedback_text,
            )

            for msg in self.convo:
                if str(msg.message_uuid) == uuid_str:
                    raw = "" if feedback_text is None else str(feedback_text)
                    msg.feedback_text = raw.strip() or None
                    break

        if not self.dead:
            try:
                data = loads(text_data)
                action = data.pop("action", None)
                logger.debug("Action: %s", action)
                if action == "append":
                    await append(data)
                    augment_start = perf_counter()
                    await database_sync_to_async(augment_conversation_with_memory)(
                        self.convo, self.user, self.db_convo.system_prompt, self.db_convo.id
                    )
                    logger.info(
                        "Memory augmentation took %.1fms in receive()",
                        (perf_counter() - augment_start) * 1000,
                    )
                    logger.debug("About to call llm_if.spin() in receive()")
                    llm_start = perf_counter()
                    await self.llm_if.spin(
                        self.convo,
                        max_func_calls=CHAT_MAX_FUNC_CALLS,
                        max_tokens=CHAT_MAX_TOKENS,
                        send_func=lambda c: self._send_conversation_delta(
                            c, create_memories=False, close_db=True
                        ),
                        stream_func=self._send_stream_payload,
                    )
                    logger.info("LLM spin took %.1fms in receive()", (perf_counter() - llm_start) * 1000)
                    await self.__save(create_memories=True)
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
                await self.send(text_data=dumps({"exception": msg}))
            except Exception as e:
                logger.error("Exception in receive(): %s", e, exc_info=True)
                if DEBUG:
                    from django.views.debug import ExceptionReporter

                    reporter = ExceptionReporter(None, *sys.exc_info())
                    debug_html = reporter.get_traceback_html()
                    await self.send(text_data=dumps({"exception": str(e), "debug_html": debug_html}))
                else:
                    await self.send(
                        text_data='{"exception": "A server error has occurred. Try reloading the page"}'
                    )


__all__ = ["ChatConsumer"]
