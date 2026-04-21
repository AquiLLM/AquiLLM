"""MCP stdio client: connects to an MCP server over stdin/stdout."""
from __future__ import annotations

import json
import os
import subprocess
import threading
from dataclasses import dataclass, field
from typing import Any

import structlog

from lib.mcp.config import MCPServerConfig

logger = structlog.stdlib.get_logger(__name__)

# Default timeout for MCP tool invocations (seconds).
MCP_TOOL_TIMEOUT = int(os.environ.get("MCP_TOOL_TIMEOUT", "30"))


@dataclass
class MCPToolSchema:
    """Schema for a tool exposed by an MCP server."""

    name: str
    description: str
    input_schema: dict[str, Any]
    server_name: str


@dataclass
class MCPClient:
    """Manages a stdio MCP server subprocess and JSON-RPC communication.

    Fully synchronous — safe to call from any thread. Uses a lock to
    serialize access to the subprocess pipes.
    """

    config: MCPServerConfig
    _process: subprocess.Popen | None = field(default=None, init=False, repr=False)
    _request_id: int = field(default=0, init=False, repr=False)
    _initialized: bool = field(default=False, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def start(self) -> None:
        """Launch the MCP server subprocess."""
        env = {**os.environ, **self.config.env}
        self._process = subprocess.Popen(
            [self.config.command, *self.config.args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        self._initialize()

    def _initialize(self) -> None:
        """Send MCP initialize handshake."""
        result = self._send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "aquillm", "version": "1.0.0"},
            },
        )
        if result is not None:
            self._send_notification("notifications/initialized", {})
            self._initialized = True

    def list_tools(self) -> list[MCPToolSchema]:
        """Discover available tools from the server."""
        if not self._initialized:
            return []
        result = self._send_request("tools/list", {})
        if result is None:
            return []
        tools: list[MCPToolSchema] = []
        for tool_data in result.get("tools", []):
            tools.append(
                MCPToolSchema(
                    name=tool_data.get("name", ""),
                    description=tool_data.get("description", ""),
                    input_schema=tool_data.get("inputSchema", {}),
                    server_name=self.config.name,
                )
            )
        return tools

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Invoke a tool on the MCP server and return the result."""
        result = self._send_request(
            "tools/call",
            {"name": name, "arguments": arguments},
        )
        if result is None:
            return {
                "isError": True,
                "content": [{"type": "text", "text": "No response from MCP server"}],
            }
        return result

    def stop(self) -> None:
        """Terminate the MCP server subprocess."""
        if self._process and self._process.returncode is None:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            except OSError:
                pass
        self._process = None
        self._initialized = False

    def _send_request(self, method: str, params: dict) -> dict | None:
        """Send a JSON-RPC request and wait for the matching response.

        Skips any server-initiated notifications received while waiting.
        """
        if not self._process or not self._process.stdin or not self._process.stdout:
            return None

        with self._lock:
            self._request_id += 1
            request_id = self._request_id
            request = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params,
            }
            payload = json.dumps(request) + "\n"
            self._process.stdin.write(payload.encode())
            self._process.stdin.flush()

            # Read lines until we get a response matching our request id,
            # skipping any server-initiated notifications.
            while True:
                line = self._process.stdout.readline()
                if not line:
                    return None

                try:
                    response = json.loads(line.decode())
                except json.JSONDecodeError:
                    logger.warning("mcp_invalid_json", server=self.config.name)
                    return None

                # Skip notifications (no "id" field)
                if "id" not in response:
                    continue

                # Skip responses for other requests
                if response["id"] != request_id:
                    continue

                break

        if "error" in response:
            logger.warning(
                "mcp_error", server=self.config.name, error=response["error"]
            )
            return None

        return response.get("result")

    def _send_notification(self, method: str, params: dict) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        if not self._process or not self._process.stdin:
            return
        with self._lock:
            notification = {
                "jsonrpc": "2.0",
                "method": method,
                "params": params,
            }
            payload = json.dumps(notification) + "\n"
            self._process.stdin.write(payload.encode())
            self._process.stdin.flush()
