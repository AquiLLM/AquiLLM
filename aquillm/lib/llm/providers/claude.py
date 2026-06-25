"""Claude (Anthropic) LLM interface."""
from typing import Optional, override

from ..types.messages import UserMessage, ToolMessage
from ..types.conversation import Conversation
from ..types.response import LLMResponse
from .base import LLMInterface


try:
    from aquillm.settings import DEBUG
except ImportError:
    DEBUG = False

if DEBUG:
    from pprint import pp


class ClaudeInterface(LLMInterface):
    """LLM interface for Anthropic's Claude models."""

    base_args: dict = {'model': 'claude-sonnet-4-6'}

    @override
    def __init__(self, anthropic_client, model_override=None):
        self.client = anthropic_client
        if model_override:
            self.base_args = {'model': model_override}

    @override
    async def get_message(self, *args, **kwargs) -> LLMResponse:
        kwargs.pop('messages_pydantic', None)
        kwargs.pop('thinking_budget', None)
        kwargs.pop('stream_callback', None)
        kwargs.pop('stream_message_uuid', None)
        msgs = kwargs.get("messages")
        sys_t = kwargs.get("system")
        mt = kwargs.get("max_tokens")
        if isinstance(msgs, list) and sys_t is not None and mt is not None:
            from lib.llm.utils.prompt_budget import apply_preflight_trim_to_message_dicts

            _, new_max = apply_preflight_trim_to_message_dicts(str(sys_t), msgs, int(mt))
            kwargs["max_tokens"] = new_max
        response = await self.client.messages.create(**kwargs)
        if DEBUG:
            print("Claude SDK Response:")
            pp(response)
        text_block = None
        tool_block = None
        content = response.content
        for block in content:
            if hasattr(block, 'input'):
                tool_block = block
            if hasattr(block, "text"):
                text_block = block

        tool_call = {
            'tool_call_id': tool_block.id,
            'tool_call_name': tool_block.name,
            'tool_call_input': tool_block.input,
        } if tool_block else {}
        
        return LLMResponse(
            text=text_block.text if text_block else None,
            tool_call=tool_call,
            stop_reason=response.stop_reason,
            input_usage=response.usage.input_tokens,
            output_usage=response.usage.output_tokens,
            model=self.base_args['model']
        )

    @override
    async def token_count(self, conversation: Conversation, new_message: Optional[str] = None) -> int:
        messages_for_bot = [message for message in conversation if not(isinstance(message, ToolMessage) and message.for_whom == 'user')]
        new_user_message = UserMessage(content=new_message) if new_message else None
        response = await self.client.messages.count_tokens(**(self.base_args | 
                                                         {'system': conversation.system,
                                                          'messages': [message.render(include={'role', 'content'}) for message in messages_for_bot + ([new_user_message] if new_user_message else [])]}))
        return response.input_tokens


__all__ = ['ClaudeInterface']
