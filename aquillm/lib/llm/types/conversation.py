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
        tool_dict = {tool.name: tool for tool in tools}
        for message in self.messages:
            if message.tools:
                message.tools = [
                    tool_dict[tool.name]
                    for tool in message.tools
                    if tool.name in tool_dict
                ]
                if not message.tools:
                    message.tool_choice = None

        if not self.messages:
            return
        last = self.messages[-1]
        if not isinstance(last, AssistantMessage) or not last.tool_call_id:
            return
        authorized_tool = tool_dict.get(last.tool_call_name or "")
        if authorized_tool is not None:
            last.tools = [authorized_tool]
            return

        # A saved tool call can outlive its permission or implementation. Record a
        # result so the assistant can finish the turn instead of waiting forever.
        name = last.tool_call_name or "unavailable_tool"
        self.messages.append(
            ToolMessage(
                tool_name=name,
                for_whom="assistant",
                content=f"Tool {name} is no longer available for this conversation.",
                arguments=last.tool_call_input or {},
                result_dict={"exception": "Tool is no longer available"},
            )
        )

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
