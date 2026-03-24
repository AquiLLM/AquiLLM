"""LLM message types for conversation handling."""
import json
import logging
from typing import Literal, Optional, Any
from os import getenv
import re
from pydantic import BaseModel, Field, model_validator
from abc import ABC
from typing import override
import uuid

from .tools import LLMTool, ToolChoice, ToolResultDict

logger = logging.getLogger(__name__)

_TOOL_ARGS_RENDER_MAX = 512


def _compact_tool_prefix(tool_name: str, arguments: Optional[dict]) -> str:
    head = f"Tool:{tool_name}"
    if arguments:
        raw = json.dumps(arguments, ensure_ascii=False, default=str, separators=(",", ":"))
        if len(raw) > _TOOL_ARGS_RENDER_MAX:
            raw = raw[:_TOOL_ARGS_RENDER_MAX] + "…"
        return f"{head}\n{raw}"
    return head


class __LLMMessage(BaseModel, ABC):
    """Base class for all LLM messages."""
    role: Literal['user', 'tool', 'assistant']
    content: str
    tools: Optional[list[LLMTool]] = None
    tool_choice: Optional[ToolChoice] = None
    rating: Literal[None, 1, 2, 3, 4, 5] = None
    feedback_text: Optional[str] = None
    files: Optional[list[tuple[str, int]]] = None
    message_uuid: uuid.UUID = Field(default_factory=uuid.uuid4)
    
    @classmethod
    @model_validator(mode='after')
    def validate_tools(cls, data: Any) -> Any:
        if (data.tools and not data.tool_choice) or (data.tool_choice and not data.tools):
            raise ValueError("Both tools and tool_choice must be populated if tools are used")

    def render(self, *args, **kwargs) -> dict:
        """Render message for LLM consumption."""
        ret = self.model_dump(*args, **kwargs)
        if self.files:
            ret['content'] = ret['content'] + "\n\nFiles:\n" + "\n".join([f'name: {file[0]}, id: {file[1]}' for file in self.files])
        return ret


class UserMessage(__LLMMessage):
    """A message from the user."""
    role: Literal['user'] = 'user'


class ToolMessage(__LLMMessage):
    """A message containing tool execution results."""
    role: Literal['tool'] = 'tool'
    tool_name: str
    arguments: Optional[dict] = None
    for_whom: Literal['assistant', 'user']
    result_dict: ToolResultDict = {}

    @staticmethod
    def _sanitize_text_for_llm(text: str) -> str:
        # Keep prompt payload bounded by removing inline base64 image data URLs.
        return re.sub(
            r"data:image/[A-Za-z0-9.+-]+;base64,[A-Za-z0-9+/=\s]+",
            "[image data url redacted for context budget]",
            text or "",
            flags=re.IGNORECASE,
        )
    
    def has_images(self) -> bool:
        """Check if this tool result contains images for multimodal processing."""
        inline_images_enabled = (getenv("LLM_TOOL_INLINE_IMAGES", "0") or "").strip().lower() in {"1", "true", "yes", "on"}
        return inline_images_enabled and bool(self.result_dict and self.result_dict.get("_images"))
    
    def get_images(self) -> list[dict]:
        """Get image data from the tool result for multimodal LLM processing."""
        if not self.result_dict:
            return []
        return self.result_dict.get("_images", [])
    
    def render_multimodal_content(self) -> list[dict]:
        """
        Render content as a list of content parts for multimodal LLMs.
        Returns list of dicts with 'type' and content fields.
        """
        content_parts = []
        
        prefix = _compact_tool_prefix(self.tool_name, self.arguments)
        text_content = f"{prefix}\n{self._sanitize_text_for_llm(self.content)}"
        
        if self.result_dict and self.result_dict.get("_image_instruction"):
            text_content += f"\n\n{self.result_dict['_image_instruction']}"
        
        content_parts.append({"type": "text", "text": text_content})
        
        for img in self.get_images():
            if img.get("image_data_url"):
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": img["image_data_url"]},
                })
                if img.get("caption"):
                    content_parts.append({
                        "type": "text",
                        "text": f"[Image {img.get('result_index', '?')}: {img.get('title', 'Untitled')}] {img['caption']}"
                    })
        
        return content_parts
    
    @override
    def render(self, *args, **kwargs) -> dict:
        ret = super().render(*args, **kwargs)
        ret['role'] = 'user'  # This is what LLMs expect.
        
        sanitized = self._sanitize_text_for_llm(self.content)
        if self.has_images():
            ret['content'] = self.render_multimodal_content()
        else:
            prefix = _compact_tool_prefix(self.tool_name, self.arguments)
            ret['content'] = f"{prefix}\n{sanitized}"

        logger.info(
            "tool_message_render tool=%s content_chars=%d",
            self.tool_name,
            len(sanitized),
        )

        ret.pop('result_dict', None)
        return ret


class AssistantMessage(__LLMMessage):
    """A message from the LLM assistant."""
    role: Literal['assistant'] = 'assistant'
    model: Optional[str] = None
    stop_reason: str
    tool_call_id: Optional[str] = None
    tool_call_name: Optional[str] = None
    tool_call_input: Optional[dict] = None
    usage: int = 0

    @classmethod
    @model_validator(mode='after')
    def validate_tool_call(cls, data: Any) -> Any:
        if (any([data.tool_call_id, data.tool_call_name]) and
        not all([data.tool_call_id, data.tool_call_name])):
            raise ValueError("If a tool call is made, both tool_call_id and tool_call_name must have values")


# Union type prevents anything at runtime from constructing LLM_Messages directly.
LLM_Message = UserMessage | ToolMessage | AssistantMessage 


__all__ = ['UserMessage', 'ToolMessage', 'AssistantMessage', 'LLM_Message']
