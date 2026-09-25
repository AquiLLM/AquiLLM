from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol
from uuid import UUID

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
    "entity_mentions",
}


class ProjectionSnapshotLoader(Protocol):
    def load(
        self, *, projection_id: UUID, batch_size: int, purpose: str
    ) -> Mapping[str, object]: ...


_PURPOSE_STATES = {
    "build": frozenset(("building",)),
    "audit": frozenset(("ready",)),
    "prune": frozenset(("failed", "superseded")),
}


def _purpose(value: object) -> str:
    if type(value) is not str or value not in _PURPOSE_STATES:
        raise ValueError("projection load purpose must be build, audit, or prune")
    return value


class DjangoProjectionRowSource:
    """Encode one selected PostgreSQL graph generation into closed opaque DTOs."""

    def __init__(
        self,
        using: str,
        *,
        state_using: str | None = None,
        loader: ProjectionSnapshotLoader | None = None,
        identifier_key: bytes | None = None,
        identifier_key_version: str | None = None,
        schema_version: str | None = None,
        projection_version: str | None = None,
    ) -> None:
        if type(using) is not str or not using:
            raise ValueError("using must be a nonempty database alias")
        self.using = using
        self._loader = (
            loader
            if loader is not None
            else DjangoProjectionOrmLoader(
                using,
                state_using=using if state_using is None else state_using,
            )
        )
        self._identifier_key = identifier_key
        self._identifier_key_version = identifier_key_version
        self._schema_version = schema_version
        self._projection_version = projection_version

    def load_projection_rows(
        self, *, projection_id: UUID, batch_size: int, purpose: str = "build"
    ) -> Mapping[str, object]:
        snapshot = self._snapshot(projection_id, batch_size, _purpose(purpose))
        return encode_projection_snapshot(
            snapshot=snapshot,
            codec=self._codec(snapshot),
        )

    def load_private_chunk_rows(
        self, *, projection_id: UUID, batch_size: int
    ) -> tuple[PrivateProjectionChunkReferenceV1, ...]:
        snapshot = self._snapshot(projection_id, batch_size, "build")
        return encode_private_rows(snapshot=snapshot, codec=self._codec(snapshot))

    def _snapshot(
        self, projection_id: UUID, batch_size: int, purpose: str
    ) -> dict[str, object]:
        loaded = self._loader.load(
            projection_id=projection_id,
            batch_size=batch_size,
            purpose=purpose,
        )
        if type(loaded) is not dict or set(loaded) != _FAMILIES:
            raise ValueError("Django projection snapshot is incomplete")
        projection = loaded["projection"]
        if (
            type(projection) is not dict
            or projection.get("id") != projection_id
            or projection.get("state") not in _PURPOSE_STATES[purpose]
        ):
            raise ValueError("Django projection snapshot is stale")
        if any(type(loaded[name]) is not tuple for name in _FAMILIES - {"projection"}):
            raise TypeError("Django projection families must be exact tuples")
        if self._schema_version is not None and (
            projection.get("schema_version") != self._schema_version
            or projection.get("projection_version") != self._projection_version
        ):
            raise ValueError("Django projection version is stale")
        return loaded

    def _codec(self, snapshot: Mapping[str, object]) -> ProjectionIdentifierCodec:
        projection = snapshot["projection"]
        expected_version = projection["identifier_key_version"]
        configured_version = self._identifier_key_version
        if configured_version != expected_version:
            raise ValueError("projection identifier key version is stale")
        secret = self._identifier_key
        if secret is None:
            raise RuntimeError("projection identifier HMAC key is not configured")
        return HmacSha256ProjectionIdentifierCodec(
            secret, key_version=configured_version
        )
