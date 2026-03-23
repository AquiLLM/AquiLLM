"""
Chat Consumers - Backward Compatibility Module

This module re-exports the ChatConsumer from its new location in apps/chat/consumers/.
New code should import directly from apps.chat.consumers.
"""
from apps.chat.consumers.chat import ChatConsumer
from apps.chat.consumers.utils import (
    CHAT_MAX_FUNC_CALLS,
    CHAT_MAX_TOKENS,
    LLM_IMAGE_MAX_BYTES,
    LLM_IMAGE_MAX_DIMENSION,
    MAX_IMAGES_PER_TOOL_RESULT,
    TOOL_CHUNK_CHAR_LIMIT,
    clean_and_parse_doc_id as _clean_and_parse_doc_id,
    env_int as _env_int,
    resize_image_for_llm_context as _resize_image_for_llm_context,
    truncate_tool_text as _truncate_tool_text,
)
from apps.chat.refs import ChatRef, CollectionsRef
from apps.chat.services import tool_wiring
from lib.tools.debug.weather import get_debug_weather_tool as get_weather_func

get_document_ids_func = tool_wiring.document_list_ids_tool
get_flat_fielding_func = tool_wiring.flat_fielding_tool
get_more_context_func = tool_wiring.more_context_tool
get_point_source_detection_func = tool_wiring.point_source_detection_tool
get_search_single_document_func = tool_wiring.search_single_document_tool
get_sky_subtraction_func = tool_wiring.sky_subtraction_tool
get_vector_search_func = tool_wiring.vector_search_tool
get_whole_document_func = tool_wiring.whole_document_tool

__all__ = [
    "CHAT_MAX_FUNC_CALLS",
    "CHAT_MAX_TOKENS",
    "ChatConsumer",
    "ChatRef",
    "CollectionsRef",
    "LLM_IMAGE_MAX_BYTES",
    "LLM_IMAGE_MAX_DIMENSION",
    "MAX_IMAGES_PER_TOOL_RESULT",
    "TOOL_CHUNK_CHAR_LIMIT",
    "_clean_and_parse_doc_id",
    "_env_int",
    "_resize_image_for_llm_context",
    "_truncate_tool_text",
    "get_document_ids_func",
    "get_flat_fielding_func",
    "get_more_context_func",
    "get_point_source_detection_func",
    "get_search_single_document_func",
    "get_sky_subtraction_func",
    "get_vector_search_func",
    "get_weather_func",
    "get_whole_document_func",
]
