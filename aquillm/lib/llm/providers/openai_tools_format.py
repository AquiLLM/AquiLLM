"""OpenAI Chat Completions `tools` and `tool_choice` wire format."""

from __future__ import annotations

from os import getenv


async def transform_openai_tools(
    tools: list[dict], *, include_strict: bool = True
) -> list[dict]:
    strict_tools = include_strict and getenv(
        "OPENAI_TOOL_STRICT", "0"
    ).strip().lower() in ("1", "true", "yes", "on")
    transformed: list[dict] = []
    for tool in tools:
        function_payload = {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": {
                "type": "object",
                "properties": tool["input_schema"]["properties"],
                # Strict wire schemas require every property. The model must
                # choose optional booleans explicitly; Python defaults remain.
                "required": (
                    list(tool["input_schema"]["properties"])
                    if strict_tools
                    else tool["input_schema"].get("required", [])
                ),
                "additionalProperties": False,
            },
        }
        if strict_tools:
            function_payload["strict"] = True
        transformed.append(
            {
                "type": "function",
                "function": function_payload,
            }
        )
    return transformed


def transform_openai_tool_choice(tool_choice: dict | None) -> str | dict | None:
    if not tool_choice:
        return None
    choice_type = tool_choice.get("type")
    if choice_type == "auto":
        return "auto"
    if choice_type == "any":
        return "required"
    if choice_type == "tool" and tool_choice.get("name"):
        return {
            "type": "function",
            "function": {"name": tool_choice["name"]},
        }
    return None


__all__ = ["transform_openai_tool_choice", "transform_openai_tools"]
