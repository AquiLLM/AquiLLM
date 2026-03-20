"""LLM response type."""
from typing import Optional
from pydantic import BaseModel


class LLMResponse(BaseModel):
    """Standardized response from an LLM provider."""
    text: Optional[str]
    tool_call: Optional[dict]
    stop_reason: str
    input_usage: int
    output_usage: int
    model: Optional[str] = None
    message_uuid: Optional[str] = None


__all__ = ['LLMResponse']
