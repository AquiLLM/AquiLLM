"""LLM tool types and utilities."""
from typing import Literal, Optional, Callable, Any
from pydantic import BaseModel, model_validator


type ToolResultValue = (
    str
    | int
    | bool
    | float
    | dict[str, ToolResultValue | list[ToolResultValue]]
    | list[tuple[str, int]]
    | list[dict]
)
type ToolResultDict = dict[Literal['exception', 'result', 'files', '_images', '_image_instruction'], ToolResultValue]


class LLMTool(BaseModel):
    """Represents a tool that can be called by an LLM."""
    llm_definition: dict
    for_whom: Literal['user', 'assistant']
    _function: Callable[..., ToolResultDict]
    
    def __init__(self, **data):
        super().__init__(**data)
        self._function = data.get("_function")

    def __call__(self, *args, **kwargs):
        return self._function(*args, **kwargs)
    
    @property
    def name(self) -> str:
        return self.llm_definition['name']


class ToolChoice(BaseModel):
    """Specifies how the LLM should choose tools."""
    type: Literal['auto', 'any', 'tool']
    name: Optional[str] = None

    @model_validator(mode='after')
    @classmethod
    def validate_name(cls, data: Any) -> Any:
        if data.type == 'tool' and data.name is None:
            raise ValueError("name is required when type is 'tool'")
        if data.type != 'tool' and data.name is not None:
            raise ValueError("name should only be set when type is 'tool'")
        return data


def dump_tool_choice(tool_choice: Any) -> dict:
    """
    Serialize tool_choice across Pydantic versions without deprecation warnings.
    """
    if tool_choice is None:
        return {}
    if hasattr(tool_choice, "model_dump"):
        return tool_choice.model_dump(exclude_none=True)
    if hasattr(tool_choice, "dict"):
        return tool_choice.dict(exclude_none=True)
    raise TypeError(f"Unsupported tool_choice type: {type(tool_choice)!r}")


__all__ = ['ToolResultDict', 'LLMTool', 'ToolChoice', 'dump_tool_choice']
