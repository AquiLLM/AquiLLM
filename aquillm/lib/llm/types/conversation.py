"""Conversation type for managing LLM message sequences."""
from pydantic import BaseModel

from .messages import LLM_Message, UserMessage, AssistantMessage, ToolMessage
from .tools import LLMTool, ToolChoice


class Conversation(BaseModel):
    """Lossless message history, including interrupted turns.

    A user message is persisted before provider work. A failed or cancelled turn
    can therefore leave adjacent user messages after the next append. Historical
    transcripts must preserve those messages and their feedback identities.
    Message-local tool invariants are validated by the individual message types.
    """
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
            if last.tool_choice is None:
                last.tool_choice = ToolChoice(type="auto")
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



__all__ = ['Conversation']
