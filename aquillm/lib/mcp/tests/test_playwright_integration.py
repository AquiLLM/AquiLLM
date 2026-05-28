"""Integration test: connect to Playwright MCP server, discover tools, invoke one."""
import asyncio
import pytest

from lib.mcp.client import MCPClient
from lib.mcp.config import MCPServerConfig
from lib.mcp.adapter import mcp_tools_to_llm_tools


@pytest.fixture
def playwright_config():
    return MCPServerConfig(
        name="playwright",
        command="npx",
        args=["@playwright/mcp@latest", "--headless"],
    )


@pytest.mark.asyncio
async def test_playwright_mcp_tool_discovery(playwright_config):
    """Start Playwright MCP, list tools, verify we get browser tools."""
    client = MCPClient(config=playwright_config)
    try:
        await client.start()
        tools = await client.list_tools()
        assert len(tools) > 0, "Playwright MCP should expose at least one tool"

        tool_names = [t.name for t in tools]
        # Playwright MCP should have browser navigation tools
        assert any("navigate" in name for name in tool_names), (
            f"Expected a navigate tool, got: {tool_names}"
        )
        print(f"\nDiscovered {len(tools)} tools: {tool_names}")
    finally:
        await client.stop()


@pytest.mark.asyncio
async def test_playwright_mcp_llm_tool_adapter(playwright_config):
    """Verify MCP tools convert to LLMTool instances correctly."""
    client = MCPClient(config=playwright_config)
    try:
        await client.start()
        schemas = await client.list_tools()
        llm_tools = mcp_tools_to_llm_tools(schemas, client)

        assert len(llm_tools) == len(schemas)
        for tool in llm_tools:
            assert tool.name.startswith("mcp_playwright_")
            assert tool.llm_definition["description"]
            assert tool.llm_definition["input_schema"]
        print(f"\nConverted {len(llm_tools)} LLMTool instances")
    finally:
        await client.stop()


@pytest.mark.asyncio
async def test_playwright_mcp_navigate(playwright_config):
    """Actually invoke the navigate tool to load a page."""
    client = MCPClient(config=playwright_config)
    try:
        await client.start()
        result = await client.call_tool(
            "browser_navigate", {"url": "https://example.com"}
        )
        assert result is not None
        assert not result.get("isError", False), f"Tool returned error: {result}"

        # Should get page content back
        content = result.get("content", [])
        assert len(content) > 0, "Expected content from navigation"
        print(f"\nNavigation result: {content[0].get('text', '')[:200]}")
    finally:
        await client.stop()
