from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol
from uuid import UUID

from django.conf import settings

from .django_projection_rows import DjangoProjectionOrmLoader
from .identifiers import (
    HmacSha256ProjectionIdentifierCodec,
    ProjectionIdentifierCodec,
)
from .projection_encoding import encode_projection_snapshot
from .projection_private_encoding import encode_private_rows
from .records import PrivateProjectionChunkReferenceV1

_FAMILIES = {
    "projection",
    "artifacts",
    "entities",
    "memberships",
    "documents",
    "chunks",
    "relations",
    "evidence",
}


class ProjectionSnapshotLoader(Protocol):
    def load(self, *, projection_id: UUID, batch_size: int) -> Mapping[str, object]: ...


def _secret_bytes(value: object) -> bytes:
    if hasattr(value, "get_secret_value"):
        value = value.get_secret_value()
    if type(value) is not str or not value:
        raise RuntimeError("projection identifier HMAC key is not configured")
    return value.encode("utf-8")


class DjangoProjectionRowSource:
    """Encode one selected PostgreSQL graph generation into closed opaque DTOs."""

    def __init__(
        self,
        using: str,
        *,
        loader: ProjectionSnapshotLoader | None = None,
        identifier_key: bytes | None = None,
        identifier_key_version: str | None = None,
    ) -> None:
        if type(using) is not str or not using:
            raise ValueError("using must be a nonempty database alias")
        self.using = using
        self._loader = (
            loader if loader is not None else DjangoProjectionOrmLoader(using)
        )
        self._identifier_key = identifier_key
        self._identifier_key_version = identifier_key_version

    def load_projection_rows(
        self, *, projection_id: UUID, batch_size: int
    ) -> Mapping[str, object]:
        snapshot = self._snapshot(projection_id, batch_size)
        return encode_projection_snapshot(
            snapshot=snapshot,
            codec=self._codec(snapshot),
        )

    def load_private_chunk_rows(
        self, *, projection_id: UUID, batch_size: int
    ) -> tuple[PrivateProjectionChunkReferenceV1, ...]:
        snapshot = self._snapshot(projection_id, batch_size)
        return encode_private_rows(snapshot=snapshot, codec=self._codec(snapshot))

    def _snapshot(self, projection_id: UUID, batch_size: int) -> dict[str, object]:
        loaded = self._loader.load(projection_id=projection_id, batch_size=batch_size)
        if type(loaded) is not dict or set(loaded) != _FAMILIES:
            raise ValueError("Django projection snapshot is incomplete")
        projection = loaded["projection"]
        if type(projection) is not dict or (
            projection.get("id"),
            projection.get("state"),
        ) != (projection_id, "building"):
            raise ValueError("Django projection snapshot is stale")
        if any(type(loaded[name]) is not tuple for name in _FAMILIES - {"projection"}):
            raise TypeError("Django projection families must be exact tuples")
        return loaded

    def _codec(self, snapshot: Mapping[str, object]) -> ProjectionIdentifierCodec:
        projection = snapshot["projection"]
        expected_version = projection["identifier_key_version"]
        configured_version = self._identifier_key_version
        if configured_version is None:
            configured_version = getattr(
                settings, "KG_PROJECTION_IDENTIFIER_KEY_VERSION", ""
            )
        if configured_version != expected_version:
            raise ValueError("projection identifier key version is stale")
        secret = self._identifier_key
        if secret is None:
            secret = _secret_bytes(
                getattr(settings, "KG_PROJECTION_IDENTIFIER_HMAC_KEY", "")
            )
        return HmacSha256ProjectionIdentifierCodec(
            secret, key_version=configured_version
        )
