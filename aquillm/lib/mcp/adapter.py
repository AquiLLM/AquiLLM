"""Adapt MCP tool schemas to LLMTool instances for the chat runtime."""
from __future__ import annotations

from typing import Any

import structlog

from lib.llm.types.tools import LLMTool, ToolResultDict
from lib.mcp.client import MCPClient, MCPToolSchema

logger = structlog.stdlib.get_logger(__name__)


def mcp_tools_to_llm_tools(
    tools: list[MCPToolSchema], client: MCPClient
) -> list[LLMTool]:
    """Convert MCP tool schemas into LLMTool instances backed by the MCP client."""
    llm_tools: list[LLMTool] = []
    for tool in tools:
        llm_tool = _make_llm_tool(tool, client)
        llm_tools.append(llm_tool)
    return llm_tools


def _make_llm_tool(schema: MCPToolSchema, client: MCPClient) -> LLMTool:
    """Create a single LLMTool wrapping an MCP tool call."""
    definition = {
        "name": f"mcp_{schema.server_name}_{schema.name}",
        "description": schema.description or f"MCP tool: {schema.name}",
        "input_schema": schema.input_schema,
    }

    def invoke(**kwargs: Any) -> ToolResultDict:
        try:
            result = client.call_tool(schema.name, kwargs)
        except Exception as exc:
            logger.warning("mcp_tool_invocation_failed", tool=schema.name, error=str(exc))
            return {"exception": f"MCP tool error: {exc}"}

        return _mcp_result_to_tool_result(result)

    return LLMTool(
        llm_definition=definition,
        for_whom="assistant",
        _function=invoke,
    )


def _mcp_result_to_tool_result(result: dict[str, Any]) -> ToolResultDict:
    """Convert MCP tool/call response to ToolResultDict."""
    if result.get("isError"):
        content = result.get("content", [])
        error_text = _extract_text_content(content) or "MCP tool returned an error"
        return {"exception": error_text}

    content = result.get("content", [])
    text = _extract_text_content(content)
    if text:
        return {"result": text}
    # Fall back to raw JSON representation
    return {"result": str(content)}


def _extract_text_content(content: list[dict[str, Any]] | Any) -> str:
    """Extract text from MCP content blocks."""
    if not isinstance(content, list):
        return str(content) if content else ""
    texts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            texts.append(block.get("text", ""))
    return "\n".join(texts)
