"""
LLM module - re-exports from lib.llm for backward compatibility.

This module has been refactored. All functionality is now in lib/llm/.
This file re-exports everything for backward compatibility with existing imports.

New code should import directly from lib.llm:
    from lib.llm import UserMessage, Conversation, llm_tool
"""

# Re-export all types
from lib.llm.types import (
    ToolResultDict, 
    LLMTool, 
    ToolChoice, 
    dump_tool_choice,
    UserMessage, 
    ToolMessage, 
    AssistantMessage, 
    LLM_Message,
    Conversation,
    LLMResponse,
)

# Re-export decorator
from lib.llm.decorators import llm_tool

# Re-export providers
from lib.llm.providers import (
    LLMInterface, 
    ClaudeInterface, 
    OpenAIInterface, 
    GeminiInterface, 
    get_provider,
    gpt_enc,
)

# Legacy alias for internal function (now public)
_dump_tool_choice = dump_tool_choice


# Helper tools that were defined in this file
@llm_tool(
    for_whom='user',
    required=['message'],
    param_descs={'message': 'The message to send to the user.'}
)
def message_to_user(message: str) -> ToolResultDict:
    """
    Send a message to the user. This is used by the LLM to communicate with the user.
    """
    return {"result": message}


@llm_tool(
    for_whom='user',
    param_descs={'strings': 'A list of strings to print'},
    required=['strings']
)
def test_function(strings: list[str]) -> ToolResultDict:
    """
    Test function that prints each string from the input. 
    """
    ret = ""
    for s in strings:
        ret += s + " "
    return {"result": ret}


__all__ = [
    # Types
    'ToolResultDict', 'LLMTool', 'ToolChoice', 'dump_tool_choice', '_dump_tool_choice',
    'UserMessage', 'ToolMessage', 'AssistantMessage', 'LLM_Message',
    'Conversation',
    'LLMResponse',
    # Decorator
    'llm_tool',
    # Providers
    'LLMInterface', 'ClaudeInterface', 'OpenAIInterface', 'GeminiInterface', 
    'get_provider', 'gpt_enc',
    # Helper tools
    'message_to_user', 'test_function',
]
