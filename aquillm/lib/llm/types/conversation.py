"""Conversation type for managing LLM message sequences."""
from typing import Any
from pydantic import BaseModel, model_validator

from .messages import LLM_Message, UserMessage, AssistantMessage, ToolMessage
from .tools import LLMTool


class Conversation(BaseModel):
    """A conversation with system prompt and message history."""
    system: str
    messages: list[LLM_Message] = []

    def __len__(self):
        return len(self.messages)
    
    def __getitem__(self, index: int):
        return self.messages[index]
    
    def __iter__(self):
        return iter(self.messages)
    
    def __add__(self, other) -> 'Conversation':
        if isinstance(other, (list, Conversation)):
            return Conversation(system=self.system, messages=self.messages + list(other))
        if isinstance(other, (UserMessage, AssistantMessage, ToolMessage)):
            return Conversation(system=self.system, messages=self.messages + [other])
        return NotImplemented

    def rebind_tools(self, tools: list[LLMTool]) -> None:
        """Rebind tool functions to messages that reference them."""
        def deprecated_func(*args, **kwargs):
            return "This tool has been deprecated."
        tool_dict = {tool.name: tool for tool in tools}
        for message in self.messages:
            if message.tools:
                for tool in message.tools:
                    if tool.name in tool_dict.keys():
                        tool._function = tool_dict[tool.name]._function
                    else:
                        tool._function = deprecated_func

    @classmethod
    def get_empty_conversation(cls):
        """Get an empty conversation with the default system prompt."""
        from django.apps import apps
        return cls(system=apps.get_app_config('aquillm').system_prompt).model_dump()

    @classmethod
    @model_validator(mode='after')
    def validate_flip_flop(cls, data: Any) -> Any:
        """Validate that messages alternate between user and assistant."""
        def isUser(m: LLM_Message):
            return isinstance(m, UserMessage) or (isinstance(m, ToolMessage) and m.for_whom == 'assistant')

        for a, b in zip(data.messages, data.messages[1:]):
            if isinstance(a, AssistantMessage) and isinstance(b, AssistantMessage):
                raise ValueError("Conversation has adjacent assistant messages")
            if isUser(a) and isUser(b):
                raise ValueError("Conversation has adjacent user messages")
        return data


__all__ = ['Conversation']
