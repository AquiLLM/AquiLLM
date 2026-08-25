"""LLM response type."""

from pydantic import BaseModel


class LLMResponse(BaseModel):
    """Standardized response from an LLM provider."""

    text: str | None
    tool_call: dict | None
    stop_reason: str
    input_usage: int
    output_usage: int
    reasoning_usage: int | None = None
    model: str | None = None
    message_uuid: str | None = None


__all__ = ["LLMResponse"]
