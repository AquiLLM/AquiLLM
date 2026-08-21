"""Strict configuration for the self-hosted topology query gateway."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from ipaddress import ip_address
from urllib.parse import urlsplit

from lib.knowledge_graph.retrieval_config import SecretSetting

from .gateway_contracts import MAX_REQUEST_BYTES, MAX_RESPONSE_BYTES

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_INTERNAL_HOSTS = frozenset({"localhost", "memgraph_knowledge_graph"})


class GatewayConfigError(ValueError):
    """Raised when the gateway service configuration is unsafe."""


@dataclass(frozen=True, slots=True)
class TopologyGatewaySettings:
    bearer_token: SecretSetting = field(repr=False)
    memgraph_uri: str = field(repr=False)
    memgraph_database: str
    query_username: str
    query_password: SecretSetting = field(repr=False)
    max_request_bytes: int
    max_response_bytes: int
    timeout_ms: int


def _error(key: str, reason: str) -> GatewayConfigError:
    return GatewayConfigError(f"{key} {reason}")


def _text(
    env: Mapping[str, str], key: str, *, secret: bool = False, ascii_only: bool = False
) -> str:
    value = env.get(key)
    if type(value) is not str:
        raise _error(key, "must be an exact string")
    if (
        not value
        or len(value) > 4096
        or (not secret and value != value.strip())
        or (ascii_only and not value.isascii())
        or any(
            ord(character) < 32
            or ord(character) == 127
            or 0xD800 <= ord(character) <= 0xDFFF
            for character in value
        )
    ):
        raise _error(key, "must be nonempty bounded canonical text")
    return value


def _integer(env: Mapping[str, str], key: str, *, minimum: int, maximum: int) -> int:
    raw = _text(env, key, ascii_only=True)
    if (
        len(raw) > len(str(maximum))
        or not raw.isdecimal()
        or (len(raw) > 1 and raw[0] == "0")
    ):
        raise _error(key, "must be a canonical decimal integer")
    value = int(raw)
    if not minimum <= value <= maximum:
        raise _error(key, "is outside the supported range")
    return value


def _internal_host(host: str) -> bool:
    if host in _INTERNAL_HOSTS or host.endswith(".internal"):
        return True
    try:
        address = ip_address(host)
    except ValueError:
        return False
    return address.is_private or address.is_loopback or address.is_link_local


def _memgraph_uri(env: Mapping[str, str]) -> str:
    key = "KG_MEMGRAPH_URI"
    value = _text(env, key, ascii_only=True)
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        raise _error(key, "must be a canonical internal Bolt URL") from None
    host = parsed.hostname
    if (
        parsed.scheme != "bolt"
        or not value.startswith("bolt://")
        or host is None
        or host != host.lower()
        or not _internal_host(host)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
        or port is None
        or not 1 <= port <= 65535
        or parsed.netloc != parsed.netloc.lower()
    ):
        raise _error(key, "must be a canonical internal Bolt URL")
    return value


def load_topology_gateway_settings(
    env: Mapping[str, str],
) -> TopologyGatewaySettings:
    """Load only the exact gateway-owned Memgraph capability and resource caps."""

    if not isinstance(env, Mapping):
        raise GatewayConfigError("configuration source must be a mapping")
    database = _text(env, "KG_MEMGRAPH_DATABASE", ascii_only=True)
    username = _text(env, "KG_MEMGRAPH_QUERY_USERNAME", ascii_only=True)
    if _NAME.fullmatch(database) is None or _NAME.fullmatch(username) is None:
        raise GatewayConfigError("Memgraph database or username is invalid")
    return TopologyGatewaySettings(
        bearer_token=SecretSetting(
            _text(
                env,
                "KG_TOPOLOGY_GATEWAY_BEARER_TOKEN",
                secret=True,
                ascii_only=True,
            )
        ),
        memgraph_uri=_memgraph_uri(env),
        memgraph_database=database,
        query_username=username,
        query_password=SecretSetting(
            _text(env, "KG_MEMGRAPH_QUERY_PASSWORD", secret=True)
        ),
        max_request_bytes=_integer(
            env,
            "KG_TOPOLOGY_GATEWAY_MAX_REQUEST_BYTES",
            minimum=1,
            maximum=MAX_REQUEST_BYTES,
        ),
        max_response_bytes=_integer(
            env,
            "KG_TOPOLOGY_GATEWAY_MAX_RESPONSE_BYTES",
            minimum=1,
            maximum=MAX_RESPONSE_BYTES,
        ),
        timeout_ms=_integer(
            env, "KG_TOPOLOGY_GATEWAY_TIMEOUT_MS", minimum=10, maximum=5_000
        ),
    )


__all__ = [
    "GatewayConfigError",
    "TopologyGatewaySettings",
    "load_topology_gateway_settings",
]
