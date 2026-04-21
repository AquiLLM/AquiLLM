"""Tests for MCP stdio client."""
import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from lib.mcp.client import MCPClient, MCPToolSchema
from lib.mcp.config import MCPServerConfig


@pytest.fixture
def config():
    return MCPServerConfig(name="test", command="echo", args=["hello"])


class TestMCPClient:
    def test_initial_state(self, config):
        client = MCPClient(config=config)
        assert client._process is None
        assert client._initialized is False

    @pytest.mark.asyncio
    async def test_stop_when_not_started(self, config):
        client = MCPClient(config=config)
        await client.stop()  # Should not raise

    @pytest.mark.asyncio
    async def test_list_tools_when_not_initialized(self, config):
        client = MCPClient(config=config)
        result = await client.list_tools()
        assert result == []

    @pytest.mark.asyncio
    async def test_send_request_returns_none_without_process(self, config):
        client = MCPClient(config=config)
        result = await client._send_request("test", {})
        assert result is None


class TestMCPToolSchema:
    def test_creation(self):
        schema = MCPToolSchema(
            name="search",
            description="Search files",
            input_schema={"type": "object", "properties": {}},
            server_name="fs-server",
        )
        assert schema.name == "search"
        assert schema.server_name == "fs-server"
