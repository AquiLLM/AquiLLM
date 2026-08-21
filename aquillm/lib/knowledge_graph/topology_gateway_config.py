"""Pure web-client configuration for the internal topology gateway."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from ipaddress import ip_address
from urllib.parse import urlsplit

GATEWAY_SETTING_KEYS = frozenset(
    {
        "KG_TOPOLOGY_GATEWAY_URL",
        "KG_TOPOLOGY_GATEWAY_BEARER_TOKEN",
        "KG_TOPOLOGY_GATEWAY_TIMEOUT_MS",
        "KG_TOPOLOGY_GATEWAY_MAX_REQUEST_BYTES",
        "KG_TOPOLOGY_GATEWAY_MAX_RESPONSE_BYTES",
    }
)
_DEFAULTS = {
    "KG_TOPOLOGY_GATEWAY_URL": "",
    "KG_TOPOLOGY_GATEWAY_BEARER_TOKEN": "",
    "KG_TOPOLOGY_GATEWAY_TIMEOUT_MS": "300",
    "KG_TOPOLOGY_GATEWAY_MAX_REQUEST_BYTES": "16384",
    "KG_TOPOLOGY_GATEWAY_MAX_RESPONSE_BYTES": "1048576",
}


class GatewayClientConfigError(ValueError):
    """Raised when the web-to-gateway capability is unsafe."""


class GatewaySecretSetting:
    __slots__ = ("__value",)

    def __init__(self, value: str) -> None:
        if type(value) is not str:
            raise TypeError("gateway secret must be an exact string")
        object.__setattr__(self, "_GatewaySecretSetting__value", value)

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("gateway secret is immutable")

    def __repr__(self) -> str:
        return "<redacted>"

    __str__ = __repr__

    def __bool__(self) -> bool:
        return bool(self.__value)

    def __eq__(self, other: object) -> bool:
        return type(other) is type(self) and self.__value == other.__value

    def __hash__(self) -> int:
        return hash(self.__value)

    def __deepcopy__(self, _memo: object) -> GatewaySecretSetting:
        return self

    def get_secret_value(self) -> str:
        return self.__value


@dataclass(frozen=True, slots=True)
class TopologyGatewayClientSettings:
    url: str = field(repr=False)
    bearer_token: GatewaySecretSetting = field(repr=False)
    timeout_ms: int
    max_request_bytes: int
    max_response_bytes: int


def _error(key: str, reason: str) -> GatewayClientConfigError:
    return GatewayClientConfigError(f"{key} {reason}")


def _raw(source: Mapping[str, str], key: str) -> str:
    value = source.get(key, _DEFAULTS[key])
    if type(value) is not str:
        raise _error(key, "must be an exact string")
    if len(value) > 4096 or any(
        ord(char) < 32 or ord(char) == 127 or 0xD800 <= ord(char) <= 0xDFFF
        for char in value
    ):
        raise _error(key, "contains invalid text")
    return value


def _integer(source: Mapping[str, str], key: str, maximum: int) -> int:
    raw = _raw(source, key)
    if (
        not raw.isascii()
        or not raw.isdecimal()
        or (len(raw) > 1 and raw[0] == "0")
        or len(raw) > len(str(maximum))
    ):
        raise _error(key, "must be a canonical decimal integer")
    return int(raw)


def _internal_host(host: str) -> bool:
    if host in {"localhost", "knowledge_graph_query_gateway"} or host.endswith(
        ".internal"
    ):
        return True
    try:
        address = ip_address(host)
    except ValueError:
        return False
    return address.is_private or address.is_loopback or address.is_link_local


def _validate_url(value: str) -> None:
    if not value:
        return
    key = "KG_TOPOLOGY_GATEWAY_URL"
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        raise _error(key, "must be a canonical internal HTTP origin") from None
    host = parsed.hostname
    if (
        parsed.scheme not in {"http", "https"}
        or not value.startswith(f"{parsed.scheme}://")
        or host is None
        or host != host.lower()
        or not _internal_host(host)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
        or port == 0
        or not value.isascii()
    ):
        raise _error(key, "must be a canonical internal HTTP origin")


def load_topology_gateway_client_settings(
    source: Mapping[str, str], *, required: bool = False
) -> TopologyGatewayClientSettings:
    """Parse only the HTTP capability; no Bolt or provider setting is accepted."""

    if not isinstance(source, Mapping):
        raise GatewayClientConfigError("configuration source must be a mapping")
    url = _raw(source, "KG_TOPOLOGY_GATEWAY_URL")
    token = _raw(source, "KG_TOPOLOGY_GATEWAY_BEARER_TOKEN")
    _validate_url(url)
    if token and (not token.isascii() or token != token.strip()):
        raise _error("KG_TOPOLOGY_GATEWAY_BEARER_TOKEN", "must be canonical ASCII")
    if not url and token:
        raise _error("KG_TOPOLOGY_GATEWAY_URL", "must accompany the bearer token")
    if url and not token:
        raise _error(
            "KG_TOPOLOGY_GATEWAY_BEARER_TOKEN", "must accompany the gateway URL"
        )
    if required and not url:
        raise _error("KG_TOPOLOGY_GATEWAY_URL", "is required for traversal")
    timeout = _integer(source, "KG_TOPOLOGY_GATEWAY_TIMEOUT_MS", 5_000)
    request_cap = _integer(source, "KG_TOPOLOGY_GATEWAY_MAX_REQUEST_BYTES", 16_384)
    response_cap = _integer(source, "KG_TOPOLOGY_GATEWAY_MAX_RESPONSE_BYTES", 1_048_576)
    if not 10 <= timeout <= 5_000:
        raise _error("KG_TOPOLOGY_GATEWAY_TIMEOUT_MS", "is outside the supported range")
    if request_cap != 16_384:
        raise _error("KG_TOPOLOGY_GATEWAY_MAX_REQUEST_BYTES", "must match the wire cap")
    if response_cap != 1_048_576:
        raise _error(
            "KG_TOPOLOGY_GATEWAY_MAX_RESPONSE_BYTES", "must match the wire cap"
        )
    return TopologyGatewayClientSettings(
        url,
        GatewaySecretSetting(token),
        timeout,
        request_cap,
        response_cap,
    )


def django_topology_gateway_client_values(
    source: Mapping[str, str],
) -> dict[str, object]:
    selected = {key: source[key] for key in GATEWAY_SETTING_KEYS if key in source}
    settings = load_topology_gateway_client_settings(selected)
    return {
        "KG_TOPOLOGY_GATEWAY_URL": settings.url,
        "KG_TOPOLOGY_GATEWAY_BEARER_TOKEN": settings.bearer_token,
        "KG_TOPOLOGY_GATEWAY_TIMEOUT_MS": settings.timeout_ms,
        "KG_TOPOLOGY_GATEWAY_MAX_REQUEST_BYTES": settings.max_request_bytes,
        "KG_TOPOLOGY_GATEWAY_MAX_RESPONSE_BYTES": settings.max_response_bytes,
    }


__all__ = [
    "GATEWAY_SETTING_KEYS",
    "GatewayClientConfigError",
    "GatewaySecretSetting",
    "TopologyGatewayClientSettings",
    "django_topology_gateway_client_values",
    "load_topology_gateway_client_settings",
]
