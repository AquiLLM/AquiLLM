"""Process-local socket denial for deterministic component evaluation."""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field


class NetworkAccessError(RuntimeError):
    """Raised when evaluated component code attempts outbound network access."""


@dataclass
class NetworkAttempts:
    """Audit log populated by :func:`deny_network`."""

    details: list[dict[str, str]] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.details)


def _is_loopback(address: object) -> bool:
    if not isinstance(address, tuple) or not address:
        return False
    host = address[0]
    if not isinstance(host, str):
        return False
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@contextmanager
def deny_network(*, allow_loopback: bool = False) -> Iterator[NetworkAttempts]:
    """Deny common socket connection APIs and restore them on context exit."""
    attempts = NetworkAttempts()
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_create_connection = socket.create_connection

    def blocked(operation: str, address: object) -> None:
        attempts.details.append({"operation": operation, "address": repr(address)})
        raise NetworkAccessError(f"network access denied: {operation}")

    def connect(_socket, address):
        if allow_loopback and _is_loopback(address):
            return original_connect(_socket, address)
        return blocked("socket.socket.connect", address)

    def connect_ex(_socket, address):
        if allow_loopback and _is_loopback(address):
            return original_connect_ex(_socket, address)
        return blocked("socket.socket.connect_ex", address)

    def create_connection(address, *args, **kwargs):
        if allow_loopback and _is_loopback(address):
            return original_create_connection(address, *args, **kwargs)
        return blocked("socket.create_connection", address)

    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex
    socket.create_connection = create_connection
    try:
        yield attempts
    finally:
        socket.socket.connect = original_connect
        socket.socket.connect_ex = original_connect_ex
        socket.create_connection = original_create_connection


__all__ = ["NetworkAccessError", "NetworkAttempts", "deny_network"]
