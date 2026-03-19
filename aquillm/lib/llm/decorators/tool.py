"""LLM tool decorator for creating callable tools."""
from typing import Callable, Literal, Optional, get_type_hints
from types import GenericAlias
from functools import wraps
import inspect

from pydantic import validate_call

from ..types.tools import LLMTool, ToolResultDict


# Import DEBUG setting - will be None if Django not configured
try:
    from aquillm.settings import DEBUG
except ImportError:
    DEBUG = False


@validate_call
def llm_tool(
    for_whom: Literal['user', 'assistant'], 
    description: Optional[str] = None, 
    param_descs: dict[str, str] = {}, 
    required: list[str] = []
) -> Callable[..., LLMTool]:
    """
    Decorator to convert a function into an LLM-compatible tool with runtime type checking.
    
    Args:
        for_whom: Whether tool results are for 'user' display or 'assistant' processing
        description: Description of what the tool does
        param_descs: Dictionary of parameter descriptions
        required: List of required parameter names
    """
    @validate_call
    def decorator(func: Callable[..., ToolResultDict]) -> LLMTool:
        type_checked_func = validate_call(func)
        
        func_name = func.__name__
        func_desc = description or func.__doc__
        if func_desc is None:
            raise ValueError(f"Must provide function description for tool {func_name}")

        func_param_descs = param_descs or {}
        func_required = required or []
        
        @wraps(type_checked_func)
        def wrapper(*args, **kwargs) -> ToolResultDict:
            if DEBUG:
                print(f"{func_name} called!")
            try:
                return type_checked_func(*args, **kwargs)
            except Exception as e:
                if DEBUG:
                    raise e
                else:
                    return {"exception": str(e)}
        
        def translate_type(t: type | GenericAlias) -> dict:
            allowed_primitives = {
                str: "string",
                int: "integer",
                bool: "boolean"
            }
            if isinstance(t, GenericAlias):
                if t.__origin__ != list or len(t.__args__) != 1 or t.__args__[0] not in allowed_primitives.keys():
                    raise TypeError("Only lists of primitive types are supported for tool call containers")
                return {"type": "array", "items": translate_type(t.__args__[0])}
            return {"type": allowed_primitives[t]}

        param_types = get_type_hints(func)
        param_types.pop("return", None)
        signature_names = set(inspect.signature(func).parameters.keys())
        
        if set(param_types.keys()) != signature_names:
            raise TypeError(f"Missing type annotations for tool {func_name}")
        if set(func_param_descs.keys()) != signature_names:
            raise TypeError(f"Missing parameter descriptions for tool {func_name}")
            
        llm_definition = {
            "name": func_name,
            "description": func_desc,
            "input_schema": {
                "type": "object",
                "properties": {
                    k: translate_type(v) | {"description": func_param_descs[k]} 
                    for k, v in param_types.items()
                },
                "required": func_required
            },
        }
        
        return LLMTool(llm_definition=llm_definition, _function=wrapper, for_whom=for_whom)
    return decorator


__all__ = ['llm_tool']
