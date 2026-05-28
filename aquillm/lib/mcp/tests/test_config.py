"""Tests for MCP configuration parsing."""
import os
import json
import pytest
from unittest.mock import patch

from lib.mcp.config import get_mcp_config, MCPServerConfig


class TestGetMCPConfig:
    def test_disabled_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            assert get_mcp_config() == []

    def test_disabled_when_false(self):
        with patch.dict(os.environ, {"MCP_ENABLED": "0"}):
            assert get_mcp_config() == []

    def test_enabled_but_no_config(self):
        with patch.dict(os.environ, {"MCP_ENABLED": "1", "MCP_SERVER_CONFIG": ""}):
            assert get_mcp_config() == []

    def test_enabled_with_invalid_json(self):
        with patch.dict(os.environ, {"MCP_ENABLED": "1", "MCP_SERVER_CONFIG": "not json"}):
            assert get_mcp_config() == []

    def test_enabled_with_valid_config(self):
        config = {
            "my-server": {
                "command": "npx",
                "args": ["-y", "@mcp/server-fs", "/tmp"],
                "env": {"FOO": "bar"},
            }
        }
        with patch.dict(
            os.environ, {"MCP_ENABLED": "true", "MCP_SERVER_CONFIG": json.dumps(config)}
        ):
            result = get_mcp_config()
            assert len(result) == 1
            assert result[0].name == "my-server"
            assert result[0].command == "npx"
            assert result[0].args == ["-y", "@mcp/server-fs", "/tmp"]
            assert result[0].env == {"FOO": "bar"}

    def test_multiple_servers(self):
        config = {
            "server-a": {"command": "cmd-a", "args": ["--flag"]},
            "server-b": {"command": "cmd-b"},
        }
        with patch.dict(
            os.environ, {"MCP_ENABLED": "1", "MCP_SERVER_CONFIG": json.dumps(config)}
        ):
            result = get_mcp_config()
            assert len(result) == 2
            names = {s.name for s in result}
            assert names == {"server-a", "server-b"}

    def test_skips_entries_without_command(self):
        config = {
            "good": {"command": "cmd"},
            "bad": {"args": ["--only-args"]},
        }
        with patch.dict(
            os.environ, {"MCP_ENABLED": "1", "MCP_SERVER_CONFIG": json.dumps(config)}
        ):
            result = get_mcp_config()
            assert len(result) == 1
            assert result[0].name == "good"
