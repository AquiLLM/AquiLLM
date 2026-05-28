"""Tests for MCP-to-LLMTool adapter."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from lib.mcp.adapter import (
    mcp_tools_to_llm_tools,
    _mcp_result_to_tool_result,
    _extract_text_content,
)
from lib.mcp.client import MCPClient, MCPToolSchema
from lib.mcp.config import MCPServerConfig


def _make_schema(name="test_tool", description="A test tool", server_name="test-server"):
    return MCPToolSchema(
        name=name,
        description=description,
        input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
        server_name=server_name,
    )


class TestMCPToolsToLLMTools:
    def test_converts_schemas_to_llm_tools(self):
        client = MagicMock(spec=MCPClient)
        schemas = [_make_schema("tool_a"), _make_schema("tool_b")]
        result = mcp_tools_to_llm_tools(schemas, client)
        assert len(result) == 2
        assert result[0].name == "mcp_test-server_tool_a"
        assert result[1].name == "mcp_test-server_tool_b"

    def test_tool_definition_has_description(self):
        client = MagicMock(spec=MCPClient)
        schemas = [_make_schema(description="Search files")]
        result = mcp_tools_to_llm_tools(schemas, client)
        assert result[0].llm_definition["description"] == "Search files"

    def test_tools_are_callable(self):
        client = MagicMock(spec=MCPClient)
        schemas = [_make_schema()]
        result = mcp_tools_to_llm_tools(schemas, client)
        assert callable(result[0])

    def test_empty_list(self):
        client = MagicMock(spec=MCPClient)
        assert mcp_tools_to_llm_tools([], client) == []


class TestMCPResultToToolResult:
    def test_success_text_content(self):
        result = _mcp_result_to_tool_result(
            {"content": [{"type": "text", "text": "hello world"}]}
        )
        assert result == {"result": "hello world"}

    def test_error_result(self):
        result = _mcp_result_to_tool_result(
            {"isError": True, "content": [{"type": "text", "text": "not found"}]}
        )
        assert result == {"exception": "not found"}

    def test_error_with_no_text(self):
        result = _mcp_result_to_tool_result({"isError": True, "content": []})
        assert "exception" in result

    def test_multiple_text_blocks(self):
        result = _mcp_result_to_tool_result(
            {
                "content": [
                    {"type": "text", "text": "line 1"},
                    {"type": "text", "text": "line 2"},
                ]
            }
        )
        assert result == {"result": "line 1\nline 2"}


class TestExtractTextContent:
    def test_empty_list(self):
        assert _extract_text_content([]) == ""

    def test_non_list(self):
        assert _extract_text_content("raw string") == "raw string"

    def test_filters_non_text_blocks(self):
        content = [
            {"type": "text", "text": "keep"},
            {"type": "image", "data": "..."},
        ]
        assert _extract_text_content(content) == "keep"
