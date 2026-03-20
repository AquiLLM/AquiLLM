"""
Chat Consumers - Backward Compatibility Module

This module re-exports the ChatConsumer from its new location in apps/chat/consumers/.
New code should import directly from apps.chat.consumers.
"""
from apps.chat.consumers.chat import (
    ChatConsumer,
    CollectionsRef,
    ChatRef,
    get_vector_search_func,
    get_document_ids_func,
    get_whole_document_func,
    get_search_single_document_func,
    get_more_context_func,
    get_sky_subtraction_func,
    get_flat_fielding_func,
    get_point_source_detection_func,
    get_weather_func,
    _truncate_tool_text,
    _clean_and_parse_doc_id,
    _resize_image_for_llm_context,
    _env_int,
    CHAT_MAX_FUNC_CALLS,
    CHAT_MAX_TOKENS,
    TOOL_CHUNK_CHAR_LIMIT,
    MAX_IMAGES_PER_TOOL_RESULT,
    LLM_IMAGE_MAX_DIMENSION,
    LLM_IMAGE_MAX_BYTES,
)

__all__ = [
    'ChatConsumer',
    'CollectionsRef',
    'ChatRef',
    'get_vector_search_func',
    'get_document_ids_func',
    'get_whole_document_func',
    'get_search_single_document_func',
    'get_more_context_func',
    'get_sky_subtraction_func',
    'get_flat_fielding_func',
    'get_point_source_detection_func',
    'get_weather_func',
    '_truncate_tool_text',
    '_clean_and_parse_doc_id',
    '_resize_image_for_llm_context',
    '_env_int',
    'CHAT_MAX_FUNC_CALLS',
    'CHAT_MAX_TOKENS',
    'TOOL_CHUNK_CHAR_LIMIT',
    'MAX_IMAGES_PER_TOOL_RESULT',
    'LLM_IMAGE_MAX_DIMENSION',
    'LLM_IMAGE_MAX_BYTES',
]
