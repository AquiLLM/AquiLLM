"""Bounded, HTTP-only driver for the internal topology gateway."""

from __future__ import annotations

import socket
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from ipaddress import ip_address
from math import isfinite
from typing import Final
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from .contracts import TopologyFailureReason, TopologyQueryName, TopologyScalar
from .failures import TopologyLoadError
from .gateway_contracts import (
    MAX_RESPONSE_BYTES,
    SCHEMA_CHECKSUM,
    SCHEMA_VERSION,
    GatewayFailureReason,
    TopologyGatewayFailureV1,
    TopologyGatewayRequestV1,
    TopologyGatewaySuccessV1,
    decode_response,
    encode_request,
)

_PATH: Final = "/v1/topology/read"
_MAX_ORIGIN_LENGTH: Final = 512
_MAX_BEARER_LENGTH: Final = 4096
_MAX_TIMEOUT_SECONDS: Final = 60.0
_SCHEMA_HEADERS: Final = {
    "X-Topology-Schema-Version": SCHEMA_VERSION,
    "X-Topology-Schema-Checksum": SCHEMA_CHECKSUM,
}


class TopologyGatewayRequestError(TimeoutError):
    """A redacted, request-local gateway failure for the hybrid scheduler."""

    __slots__ = ("reason",)

    def __init__(self, reason: GatewayFailureReason):
        if reason not in {
            GatewayFailureReason.DEADLINE,
            GatewayFailureReason.RESULT_CAP,
        }:
            raise TypeError("reason must be a local gateway failure")
        self.reason = reason
        super().__init__("topology gateway request failed")


class _RejectRedirect(HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        return None


def _internal_origin(origin: str) -> str:
    if type(origin) is not str or not origin or len(origin) > _MAX_ORIGIN_LENGTH:
        raise ValueError("gateway origin is invalid")
    try:
        parts = urlsplit(origin)
        port = parts.port
    except ValueError:
        raise ValueError("gateway origin is invalid") from None
    if (
        parts.scheme not in {"http", "https"}
        or not parts.netloc
        or parts.username is not None
        or parts.password is not None
        or parts.query
        or parts.fragment
        or parts.path not in {"", "/"}
        or port == 0
    ):
        raise ValueError("gateway origin is invalid")
    host = parts.hostname
    if host is None or not _is_internal_host(host):
        raise ValueError("gateway origin is not internal")
    return urlunsplit((parts.scheme, parts.netloc, _PATH, "", ""))


def _is_internal_host(host: str) -> bool:
    if host in {"localhost", "knowledge_graph_query_gateway"} or host.endswith(
        ".internal"
    ):
        return True
    try:
        address = ip_address(host)
    except ValueError:
        return False
    return address.is_private or address.is_loopback or address.is_link_local


def _schema_failure() -> TopologyLoadError:
    return TopologyLoadError(TopologyFailureReason.BACKEND_SCHEMA_MISMATCH)


def _unavailable() -> TopologyLoadError:
    return TopologyLoadError(TopologyFailureReason.BACKEND_UNAVAILABLE)


def _failure(reason: GatewayFailureReason) -> None:
    shared = {
        GatewayFailureReason.AUTHENTICATION: (
            TopologyFailureReason.BACKEND_AUTHENTICATION
        ),
        GatewayFailureReason.UNAVAILABLE: TopologyFailureReason.BACKEND_UNAVAILABLE,
        GatewayFailureReason.SCHEMA: TopologyFailureReason.BACKEND_SCHEMA_MISMATCH,
        GatewayFailureReason.PROVENANCE: (
            TopologyFailureReason.BACKEND_PROVENANCE_MISMATCH
        ),
    }
    if reason in shared:
        raise TopologyLoadError(shared[reason])
    raise TopologyGatewayRequestError(reason)


@dataclass(frozen=True, slots=True)
class TopologyGatewayClient:
    """POST only canonical topology DTOs to one non-proxied internal endpoint."""

    origin: str = field(repr=False)
    bearer_token: str = field(repr=False)
    timeout_ceiling: float
    _endpoint: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            type(self.bearer_token) is not str
            or not self.bearer_token
            or len(self.bearer_token) > _MAX_BEARER_LENGTH
            or not self.bearer_token.isascii()
            or any(ord(char) < 0x20 or ord(char) == 0x7F for char in self.bearer_token)
        ):
            raise ValueError("gateway bearer token is invalid")
        if (
            type(self.timeout_ceiling) is not float
            or not isfinite(self.timeout_ceiling)
            or not 0.0 < self.timeout_ceiling <= _MAX_TIMEOUT_SECONDS
        ):
            raise ValueError("gateway timeout ceiling is invalid")
        object.__setattr__(self, "_endpoint", _internal_origin(self.origin))

    def execute_read(
        self,
        *,
        query: TopologyQueryName,
        parameters: Mapping[str, TopologyScalar],
        deadline: float,
        max_records: int,
    ) -> tuple[Mapping[str, TopologyScalar], ...]:
        request_dto = TopologyGatewayRequestV1(query, parameters, deadline, max_records)
        remaining = deadline - time.monotonic()
        if not isfinite(remaining) or remaining <= 0.0:
            raise TopologyGatewayRequestError(GatewayFailureReason.DEADLINE)
        request = Request(
            self._endpoint,
            data=encode_request(request_dto),
            headers={
                "Authorization": f"Bearer {self.bearer_token}",
                "Content-Type": "application/json",
                **_SCHEMA_HEADERS,
            },
            method="POST",
        )
        opener = build_opener(ProxyHandler({}), _RejectRedirect())
        remaining = deadline - time.monotonic()
        if not isfinite(remaining) or remaining <= 0.0:
            raise TopologyGatewayRequestError(GatewayFailureReason.DEADLINE)
        timeout = min(remaining, self.timeout_ceiling)
        try:
            response = opener.open(request, timeout=timeout)
        except HTTPError as error:
            if 300 <= error.code < 400:
                try:
                    error.close()
                except OSError:
                    pass
                raise _unavailable() from None
            response = error
        except TimeoutError:
            raise TopologyGatewayRequestError(GatewayFailureReason.DEADLINE) from None
        except URLError as error:
            if isinstance(error.reason, (socket.timeout, TimeoutError)):
                raise TopologyGatewayRequestError(
                    GatewayFailureReason.DEADLINE
                ) from None
            raise _unavailable() from None
        except OSError:
            raise _unavailable() from None
        try:
            decoded = self._decode(response)
        except (TopologyLoadError, TopologyGatewayRequestError):
            raise
        except TimeoutError:
            raise TopologyGatewayRequestError(GatewayFailureReason.DEADLINE) from None
        except OSError:
            raise _unavailable() from None
        except (TypeError, ValueError):
            raise _schema_failure() from None
        finally:
            try:
                response.close()
            except OSError:
                pass
        if type(decoded) is TopologyGatewaySuccessV1:
            return decoded.rows
        _failure(decoded.reason)
        raise AssertionError("unreachable")

    def _decode(self, response) -> TopologyGatewaySuccessV1 | TopologyGatewayFailureV1:
        headers = response.headers
        if (
            headers.get("Transfer-Encoding") is not None
            or headers.get("Content-Type") != "application/json"
            or headers.get("X-Topology-Schema-Version") != SCHEMA_VERSION
            or headers.get("X-Topology-Schema-Checksum") != SCHEMA_CHECKSUM
        ):
            raise _schema_failure()
        length_text = headers.get("Content-Length")
        if (
            type(length_text) is not str
            or not length_text.isascii()
            or not length_text.isdecimal()
        ):
            raise _schema_failure()
        length = int(length_text)
        if str(length) != length_text or length > MAX_RESPONSE_BYTES:
            raise _schema_failure()
        body = response.read(length + 1)
        if type(body) is not bytes or len(body) != length:
            raise _schema_failure()
        decoded = decode_response(body)
        if type(response.status) is not int:
            raise _schema_failure()
        if type(decoded) is TopologyGatewaySuccessV1:
            if response.status != 200:
                raise _schema_failure()
        elif response.status != decoded.status:
            raise _schema_failure()
        return decoded


__all__ = ["TopologyGatewayClient", "TopologyGatewayRequestError"]
