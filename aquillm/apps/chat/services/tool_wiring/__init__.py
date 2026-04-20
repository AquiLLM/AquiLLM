"""Chat LLM tool factories: Django wiring over lib.tools implementations."""
from __future__ import annotations

from django.contrib.auth.models import User

from aquillm.llm import LLMTool
from apps.chat.refs import ChatRef, CollectionsRef

from .astronomy import (
    flat_fielding_tool,
    point_source_detection_tool,
    sky_subtraction_tool,
)
from .documents import (
    document_list_ids_tool,
    more_context_tool,
    search_single_document_tool,
    vector_search_tool,
    whole_document_tool,
)


def build_document_tools(user: User, col_ref: CollectionsRef, chat_ref: ChatRef) -> list[LLMTool]:
    return [
        vector_search_tool(user, col_ref),
        more_context_tool(user),
        document_list_ids_tool(user, col_ref),
        whole_document_tool(user, chat_ref, col_ref),
        search_single_document_tool(user, col_ref),
    ]


def build_astronomy_tools(chat_consumer: "ChatConsumer") -> list[LLMTool]:
    return [
        sky_subtraction_tool(chat_consumer),
        flat_fielding_tool(chat_consumer),
        point_source_detection_tool(chat_consumer),
    ]


__all__ = [
    "build_astronomy_tools",
    "build_document_tools",
    "document_list_ids_tool",
    "flat_fielding_tool",
    "more_context_tool",
    "point_source_detection_tool",
    "search_single_document_tool",
    "sky_subtraction_tool",
    "vector_search_tool",
    "whole_document_tool",
]
