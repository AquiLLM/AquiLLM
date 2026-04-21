"""MCP (Model Context Protocol) integration for AquiLLM."""
from lib.mcp.config import MCPServerConfig, get_mcp_config
from lib.mcp.client import MCPClient
from lib.mcp.adapter import mcp_tools_to_llm_tools

__all__ = [
    "MCPServerConfig",
    "get_mcp_config",
    "MCPClient",
    "mcp_tools_to_llm_tools",
]
