"""Finite, context-propagating synchronous tool execution."""

from functools import partial
from typing import Any, Literal

from lib.llm.turn_context import check_turn_active, submit_tool, tool_timeout
from lib.llm.types.messages import AssistantMessage, ToolMessage
from lib.llm.utils.tool_call_kwargs import normalize_tool_call_kwargs

from . import image_context as imgctx

try:
    from aquillm.settings import DEBUG
except ImportError:
    DEBUG = False


def call_tool(self, message: AssistantMessage) -> ToolMessage:
    """Execute a tool call from an assistant message."""
    tools = message.tools
    if tools:
        name = message.tool_call_name
        input = message.tool_call_input
        tools_dict = {tool.llm_definition["name"]: tool for tool in tools}
        tool_name = name or "invalid_tool"
        for_whom: Literal["assistant", "user"] = "assistant"
        result_dict: dict = {"exception": "Tool call failed before execution"}
        call_arguments: Any = input
        if not name or name not in tools_dict.keys():
            result_dict = {"exception": "Function name is not valid"}
            result = imgctx.serialize_tool_result_for_llm(result_dict)
        else:
            tool = tools_dict[name]
            tool_name = tool.name
            for_whom = tool.for_whom
            # Empty argument objects still require ordinary tool validation.
            if not isinstance(input, dict):
                result_dict = {
                    "exception": (
                        "The model returned a tool call without a JSON argument "
                        "object. "
                        "Required parameters were not supplied; try again or "
                        "simplify the request."
                    ),
                }
                result = imgctx.serialize_tool_result_for_llm(result_dict)
            else:
                call_arguments = normalize_tool_call_kwargs(name, input)
                future = submit_tool(
                    self.tool_executor, partial(tool, **call_arguments)
                )
                try:
                    tool_timeout_s = tool_timeout()
                    result_dict = future.result(timeout=tool_timeout_s)
                    check_turn_active()
                    result = imgctx.serialize_tool_result_for_llm(result_dict)
                except TimeoutError:
                    future.cancel()
                    result_dict = {"exception": "Tool call timed out"}
                    result = imgctx.serialize_tool_result_for_llm(result_dict)
                except Exception as e:
                    if DEBUG:
                        raise
                    result_dict = {"exception": str(e)}
                    result = imgctx.serialize_tool_result_for_llm(result_dict)
        return ToolMessage(
            tool_name=tool_name,
            content=result,
            arguments=call_arguments,
            result_dict=result_dict,
            for_whom=for_whom,
            tools=message.tools,
            files=result_dict.get("files") if isinstance(result_dict, dict) else None,
            tool_choice=message.tool_choice,
        )
    raise ValueError("call_tool called on a message with no tools!")
