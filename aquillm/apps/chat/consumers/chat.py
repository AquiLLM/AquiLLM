"""WebSocket consumer for chat functionality."""

from __future__ import annotations

from json import dumps
from os import getenv
from time import perf_counter
from typing import Any

import structlog
from anthropic._exceptions import OverloadedError
from channels.db import database_sync_to_async
from channels.generic.websocket import AsyncWebsocketConsumer
from django.apps import apps
from django.contrib.auth.models import User

from apps.chat.consumers.chat_delta import (
    send_conversation_delta,
    send_conversation_snapshot,
)
from apps.chat.consumers.chat_publish import run_llm_spin
from apps.chat.consumers.chat_receive import (
    handle_chat_receive,
    restore_pending_user_tool_intent,
)
from apps.chat.consumers.chat_transport import best_effort_send
from apps.chat.consumers.chat_ws_errors import send_connect_error
from apps.chat.consumers.utils import CHAT_MAX_FUNC_CALLS, CHAT_MAX_TOKENS
from apps.chat.models import WSConversation
from apps.chat.refs import ChatRef, CollectionsRef
from apps.chat.services.execution import execution_turn
from apps.chat.services.rag_pipeline import run_direct_rag_turn
from apps.chat.services.rag_turn import preservation_turn
from apps.chat.services.skills_runtime import (
    build_skill_tools,
    effective_base_system_for_memory_async,
)
from apps.chat.services.tool_wiring import (
    build_astronomy_tools,
    build_document_tools,
    build_memory_tools,
)
from apps.chat.tasks import enqueue_index_conversation_task
from aquillm.llm import LLMInterface, LLMTool, message_to_user
from aquillm.memory import augment_conversation_with_memory_async
from aquillm.message_adapters import (
    ConversationConflictError,
    load_conversation_from_db,
)
from aquillm.settings import DEBUG, SKILLS_ENABLED
from aquillm.tasks import enqueue_conversation_memories_task
from lib.tools.debug.weather import get_debug_weather_tool

logger = structlog.stdlib.get_logger(__name__)


class ChatConsumer(AsyncWebsocketConsumer):
    llm_if: LLMInterface = apps.get_app_config("aquillm").llm_interface
    db_convo: WSConversation | None = None
    convo: Any | None = None
    tools: list[LLMTool] = []
    memory_tools: list[LLMTool] = []
    user: User | None = None

    dead: bool = False

    col_ref: CollectionsRef
    last_sent_sequence: int = -1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.col_ref = CollectionsRef([])
        self.transport_connected = True

    async def _send_stream_payload(self, payload: dict) -> None:
        await best_effort_send(
            self,
            text_data=dumps({"stream": payload}),
        )

    async def dispatch(self, message):
        from apps.chat.consumers.chat_turn_dispatch import dispatch_chat_event

        await dispatch_chat_event(self, message, super().dispatch)

    async def __call__(self, scope, receive, send):
        from apps.chat.consumers.chat_turn_dispatch import stop_chat_work

        try:
            await super().__call__(scope, receive, send)
        finally:
            if getattr(self, "_chat_event_owner", None):
                await stop_chat_work(self)

    async def disconnect(self, close_code):
        from apps.chat.services.rag_turn import cancel_active_turn

        cancel_active_turn(self)
        self.transport_connected = False

    async def _save_conversation(
        self, create_memories=False, *, selected_collections=None
    ):
        from apps.chat.consumers.chat_persistence import save_chat_conversation

        await save_chat_conversation(
            self,
            create_memories=create_memories,
            selected_collections=selected_collections,
            enqueue_functions=(
                enqueue_conversation_memories_task,
                enqueue_index_conversation_task,
            ),
        )

    @database_sync_to_async
    def __get_convo(self, convo_id: int, user: User):
        convo = WSConversation.objects.filter(id=convo_id).first()
        if convo:
            if convo.owner == user:
                return convo
            return None
        return convo

    def _apply_saved_collection_selection(self) -> None:
        if self.db_convo is None:
            return
        self.col_ref.collections = list(self.db_convo.selected_collection_ids or [])

    async def connect(self):
        logger.debug("obs.chat.connect")

        self.transport_connected = True
        self._chat_accepted = False
        self.dead = False
        try:
            self.user = self.scope.get("user")
            if self.user is None or not self.user.is_authenticated:
                await send_connect_error(
                    self,
                    PermissionError("Authentication required"),
                    code=4401,
                    message="Authentication required",
                )
                return
            logger.debug(
                "obs.chat.user_resolved", user_id=getattr(self.user, "id", None)
            )
            convo_id = self.scope["url_route"]["kwargs"]["convo_id"]
            logger.debug("obs.chat.convo_id_resolved", conversation_id=convo_id)
            self.db_convo = await self.__get_convo(convo_id, self.user)
            if self.db_convo is None:
                logger.error("obs.chat.invalid_conversation", conversation_id=convo_id)
                await send_connect_error(
                    self,
                    LookupError("Invalid chat_id"),
                    code=4404,
                    message="Invalid chat_id",
                )
                return

            await self.accept()
            self._chat_accepted = True
            logger.debug("obs.chat.ws_accepted")
            self._apply_saved_collection_selection()
            self.doc_tools = build_document_tools(
                self.user, self.col_ref, ChatRef(self)
            )
            self.memory_tools = build_memory_tools(self.user, ChatRef(self))
            self.tools = (
                self.doc_tools + build_astronomy_tools(self) + self.memory_tools
            )
            if getenv("LLM_CHOICE") == "GEMMA3":
                self.tools.append(message_to_user)
            if DEBUG:
                self.tools.append(get_debug_weather_tool())
            if SKILLS_ENABLED:
                self.tools = self.tools + build_skill_tools(self)
            self.convo = await database_sync_to_async(load_conversation_from_db)(
                self.db_convo
            )
            self._apply_saved_collection_selection()
            await send_conversation_snapshot(self)
            self.convo.rebind_tools(self.tools)
            if not self._has_pending_turn():
                return
            async with execution_turn(self, wait=True):
                # Another socket may have finished while this viewer waited.
                self.convo = await database_sync_to_async(load_conversation_from_db)(
                    self.db_convo
                )
                self._apply_saved_collection_selection()
                loaded_len = len(self.convo)
                await send_conversation_snapshot(self)
                self.convo.rebind_tools(self.tools)
                if not self._has_pending_turn():
                    return
                restore_pending_user_tool_intent(
                    self.convo,
                    all_tools=self.tools,
                    document_tools=self.doc_tools,
                    selected_collection_ids=self.col_ref.collections,
                    memory_tools=self.memory_tools,
                )
                augment_start = perf_counter()
                await augment_conversation_with_memory_async(
                    self.convo,
                    self.user,
                    await effective_base_system_for_memory_async(self),
                    self.db_convo.id,
                    include_episodic=not bool(self.col_ref.collections),
                )
                logger.info(
                    "obs.chat.memory_augmented",
                    phase="connect",
                    duration_ms=(perf_counter() - augment_start) * 1000,
                )
                logger.debug("obs.chat.spin_starting", phase="connect")
                llm_start = perf_counter()
                async with preservation_turn(self, max_func_calls=CHAT_MAX_FUNC_CALLS):
                    direct_outcome = "skipped"
                    if self.convo[-1].role == "user":
                        direct_outcome = await run_direct_rag_turn(
                            self,
                            self.llm_if,
                            self.convo,
                            stream_func=self._send_stream_payload,
                        )
                    if direct_outcome == "handled":
                        await send_conversation_delta(
                            self,
                            self.convo,
                            create_memories=False,
                            close_db=False,
                        )
                    else:
                        await run_llm_spin(
                            self,
                            self.llm_if,
                            self.convo,
                            max_func_calls=CHAT_MAX_FUNC_CALLS,
                            max_tokens=CHAT_MAX_TOKENS,
                            send_func=lambda c: send_conversation_delta(
                                self, c, create_memories=False, close_db=False
                            ),
                            stream_func=self._send_stream_payload,
                        )
                    logger.info(
                        "obs.chat.spin_completed",
                        phase="connect",
                        duration_ms=(perf_counter() - llm_start) * 1000,
                    )
                    if len(self.convo) > loaded_len:
                        await self._save_conversation(create_memories=True)
                logger.debug("obs.chat.spin_done", phase="connect")
        except ConversationConflictError as e:
            logger.warning("obs.chat.conversation_conflict", error=str(e))
            await send_connect_error(
                self,
                e,
                code=4409,
                message=(
                    "This chat changed in another connection. Refresh the page "
                    "and resend your message if needed."
                ),
            )
        except OverloadedError as e:
            logger.error(
                "obs.chat.llm_overloaded", error=str(e), error_type=type(e).__name__
            )
            await send_connect_error(
                self,
                e,
                code=1013,
                message="LLM provider is currently overloaded. Try again later.",
            )
        except Exception as e:
            logger.error(
                "obs.chat.connect_error",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )
            await send_connect_error(self, e, code=1011)

    def _has_pending_turn(self) -> bool:
        if not self.convo:
            return False
        last = self.convo[-1]
        if last.role == "user":
            return True
        if last.role == "tool":
            return last.for_whom == "assistant"
        return bool(last.tool_call_id)

    async def receive(self, text_data):
        await handle_chat_receive(self, text_data)


__all__ = ["ChatConsumer"]
