"""
LLM library for AquiLLM.

This module provides:
- Types for messages, conversations, tools, and responses
- The @llm_tool decorator for creating LLM-callable tools
- Provider interfaces for Claude, OpenAI, and Gemini
"""
from .types import (
    ToolResultDict, LLMTool, ToolChoice, dump_tool_choice,
    UserMessage, ToolMessage, AssistantMessage, LLM_Message,
    Conversation,
    LLMResponse,
)
from .decorators import llm_tool
from .providers import (
    LLMInterface, 
    ClaudeInterface, 
    OpenAIInterface, 
    GeminiInterface, 
    get_provider,
    gpt_enc,
)


__all__ = [
    # Types
    'ToolResultDict', 'LLMTool', 'ToolChoice', 'dump_tool_choice',
    'UserMessage', 'ToolMessage', 'AssistantMessage', 'LLM_Message',
    'Conversation',
    'LLMResponse',
    # Decorators
    'llm_tool',
    # Providers
    'LLMInterface', 'ClaudeInterface', 'OpenAIInterface', 'GeminiInterface', 
    'get_provider', 'gpt_enc',
]
