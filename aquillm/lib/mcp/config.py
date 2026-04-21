"""MCP server configuration parsing."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field


@dataclass
class MCPServerConfig:
    """Configuration for a single MCP server (stdio transport)."""

    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)


def get_mcp_config() -> list[MCPServerConfig]:
    """Parse MCP server configuration from environment.

    Reads MCP_SERVER_CONFIG as a JSON object mapping server names to their
    stdio transport settings:

        {
            "my-server": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
                "env": {}
            }
        }

    Returns empty list if MCP_ENABLED is falsy or config is missing/invalid.
    """
    if not os.environ.get("MCP_ENABLED", "").strip().lower() in ("1", "true", "yes"):
        return []

    raw = os.environ.get("MCP_SERVER_CONFIG", "").strip()
    if not raw:
        return []

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []

    if not isinstance(data, dict):
        return []

    servers: list[MCPServerConfig] = []
    for name, cfg in data.items():
        if not isinstance(cfg, dict) or "command" not in cfg:
            continue
        servers.append(
            MCPServerConfig(
                name=str(name),
                command=str(cfg["command"]),
                args=[str(a) for a in cfg.get("args", [])],
                env={str(k): str(v) for k, v in cfg.get("env", {}).items()},
            )
        )
    return servers
