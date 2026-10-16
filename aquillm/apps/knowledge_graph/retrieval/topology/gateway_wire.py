"""Bounded ASGI request and response encoding shared by topology routes."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from .gateway_contracts import (
    SCHEMA_CHECKSUM,
    SCHEMA_VERSION,
    GatewayFailureReason,
    TopologyGatewayFailureV1,
    encode_response,
)


def _header_values(scope: dict[str, Any], name: bytes) -> tuple[bytes, ...]:
    return tuple(
        value
        for key, value in scope.get("headers", ())
        if type(key) is bytes and type(value) is bytes and key.lower() == name
    )


def _single_header(scope: dict[str, Any], name: bytes) -> bytes | None:
    values = _header_values(scope, name)
    return values[0] if len(values) == 1 else None


def _response_headers(
    body: bytes,
    *,
    version=SCHEMA_VERSION,
    checksum=SCHEMA_CHECKSUM,
    content_type=b"application/json",
) -> list[tuple[bytes, bytes]]:
    return [
        (b"content-type", content_type),
        (b"content-length", str(len(body)).encode()),
        (b"x-topology-schema-version", version.encode()),
        (b"x-topology-schema-checksum", checksum.encode()),
    ]


async def _respond(
    send: Callable[..., Awaitable[None]],
    status: int,
    body: bytes,
    *,
    version=SCHEMA_VERSION,
    checksum=SCHEMA_CHECKSUM,
    content_type=b"application/json",
) -> None:
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": _response_headers(
                body, version=version, checksum=checksum, content_type=content_type
            ),
        }
    )
    await send({"type": "http.response.body", "body": body})


async def _failure(
    send: Callable[..., Awaitable[None]],
    reason: GatewayFailureReason,
    *,
    version=SCHEMA_VERSION,
    checksum=SCHEMA_CHECKSUM,
) -> None:
    value = TopologyGatewayFailureV1(reason)
    await _respond(
        send,
        value.status,
        encode_response(value),
        version=version,
        checksum=checksum,
    )


async def _read_body(
    receive: Callable[..., Awaitable[dict[str, Any]]], expected: int
) -> bytes | None:
    chunks: list[bytes] = []
    size = 0
    while True:
        message = await receive()
        if message.get("type") != "http.request":
            return None
        chunk = message.get("body", b"")
        if type(chunk) is not bytes:
            return None
        size += len(chunk)
        if size > expected:
            return None
        chunks.append(chunk)
        if not message.get("more_body", False):
            body = b"".join(chunks)
            return body if len(body) == expected else None


def _wire_valid(
    scope: dict[str, Any], *, version=SCHEMA_VERSION, checksum=SCHEMA_CHECKSUM
) -> bool:
    return (
        _single_header(scope, b"content-type") == b"application/json"
        and _single_header(scope, b"x-topology-schema-version") == version.encode()
        and _single_header(scope, b"x-topology-schema-checksum") == checksum.encode()
        and not _header_values(scope, b"transfer-encoding")
    )


def _length(scope: dict[str, Any], maximum: int) -> int | None:
    raw = _single_header(scope, b"content-length")
    if raw is None or not raw.isascii():
        return None
    text = raw.decode("ascii")
    if not text.isdecimal() or (len(text) > 1 and text[0] == "0"):
        return None
    return maximum + 1 if len(text) > len(str(maximum)) else int(text)
