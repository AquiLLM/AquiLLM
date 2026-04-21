"""WebSocket consumer for chat functionality."""
from __future__ import annotations

import asyncio

import structlog
from json import dumps
from os import getenv
from time import perf_counter
from typing import Any, Optional

from anthropic._exceptions import OverloadedError
from channels.db import database_sync_to_async
from channels.generic.websocket import AsyncWebsocketConsumer
from django.apps import apps
from django.contrib.auth.models import User

from aquillm.llm import LLMInterface, LLMTool, message_to_user
from aquillm.memory import augment_conversation_with_memory_async
from aquillm.message_adapters import load_conversation_from_db, pydantic_message_to_frontend_dict
from aquillm.settings import DEBUG
from aquillm.tasks import enqueue_conversation_memories_task
from apps.chat.consumers.chat_delta import send_conversation_delta
from apps.chat.consumers.chat_receive import handle_chat_receive
from apps.chat.consumers.chat_ws_errors import send_connect_error
from apps.chat.consumers.utils import CHAT_MAX_FUNC_CALLS, CHAT_MAX_TOKENS
from apps.chat.models import WSConversation
from apps.chat.refs import ChatRef, CollectionsRef
from apps.chat.services.tool_wiring import build_astronomy_tools, build_document_tools
from lib.mcp.config import get_mcp_config
from lib.mcp.client import MCPClient
from lib.mcp.adapter import mcp_tools_to_llm_tools
from lib.tools.debug.weather import get_debug_weather_tool

logger = structlog.stdlib.get_logger(__name__)


class ChatConsumer(AsyncWebsocketConsumer):
    llm_if: LLMInterface = apps.get_app_config("aquillm").llm_interface
    db_convo: Optional[WSConversation] = None
    convo: Optional[Any] = None
    tools: list[LLMTool] = []
    user: Optional[User] = None

    dead: bool = False
    _mcp_clients: list[MCPClient]

    col_ref = CollectionsRef([])
    last_sent_sequence: int = -1

    async def _send_stream_payload(self, payload: dict) -> None:
        await self.send(text_data=dumps({"stream": payload}))

    @database_sync_to_async
    def _save_conversation(self, create_memories: bool = False):
        from aquillm.message_adapters import save_conversation_to_db

        assert self.db_convo is not None
        save_conversation_to_db(self.convo, self.db_convo)
        if create_memories:
            try:
                enqueue_conversation_memories_task(
                    conversation_id=self.db_convo.id,
                    queued_updated_at=self.db_convo.updated_at.isoformat(),
                )
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
        # MCP tool discovery (stdio transport)
        self._mcp_clients = []
        mcp_configs = get_mcp_config()
        logger.info("mcp_discovery_start", enabled=bool(mcp_configs), server_count=len(mcp_configs))
        for cfg in mcp_configs:
            try:
                client = MCPClient(config=cfg)
                await asyncio.to_thread(client.start)
                schemas = await asyncio.to_thread(client.list_tools)
                self.tools.extend(mcp_tools_to_llm_tools(schemas, client))
                self._mcp_clients.append(client)
                logger.info("mcp_server_connected", server=cfg.name, tool_count=len(schemas))
            except Exception as exc:
                logger.warning("mcp_server_failed", server=cfg.name, error=str(exc))
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
            await augment_conversation_with_memory_async(
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
                send_func=lambda c: send_conversation_delta(
                    self, c, create_memories=False, close_db=False
                ),
                stream_func=self._send_stream_payload,
            )
            logger.info("LLM spin took %.1fms in connect()", (perf_counter() - llm_start) * 1000)
            await self._save_conversation(create_memories=len(self.convo) > before_spin_len)
            logger.debug("llm_if.spin() completed in connect()")
            return
        except OverloadedError as e:
            logger.error("LLM overloaded: %s", e)
            self.dead = True
            await self.send('{"exception": "LLM provider is currently overloaded. Try again later."}')
            return
        except Exception as e:
            logger.error("Exception in connect(): %s", e, exc_info=True)
            await send_connect_error(self, e)
            return

    async def disconnect(self, code):
        for client in getattr(self, "_mcp_clients", []):
            try:
                await asyncio.to_thread(client.stop)
            except Exception as exc:
                logger.debug("mcp_client_cleanup_error", error=str(exc))

    async def receive(self, text_data):
        await handle_chat_receive(self, text_data)


__all__ = ["ChatConsumer"]
