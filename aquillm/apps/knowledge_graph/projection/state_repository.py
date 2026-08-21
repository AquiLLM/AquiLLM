# ruff: noqa: E501,E701,E702,I001
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from django.db import connections
from django.utils import timezone

from apps.knowledge_graph.models import CollectionGraphProjection, ProjectionChunkReference

from .records import PrivateProjectionChunkReferenceV1, ProjectionFailureCode

_FUNCTIONS = {
    "claim": "kg_projection_claim",
    "renew": "kg_projection_renew",
    "record_mapping": "kg_projection_record_private_mapping",
    "fail": "kg_projection_fail",
    "supersede": "kg_projection_supersede",
    "store_chunks": "kg_projection_store_chunk_references",
    "fence_chunks": "kg_projection_fence_chunk_references",
    "claim_outbox": "kg_projection_claim_outbox",
    "complete_outbox": "kg_projection_complete_outbox",
    "fail_outbox": "kg_projection_fail_outbox",
    "ready": "kg_projection_ready_compare_and_set",
    "replay": "kg_projection_replay",
}


@dataclass(frozen=True, slots=True)
class StateLeaseV1:
    projection_id: UUID
    owner: str
    expires_at: datetime
    attempt_count: int


@dataclass(frozen=True, slots=True)
class StateReadyOutcomeV1:
    projection_id: UUID
    published: bool
    state: str
    failure_code: str | None


@dataclass(frozen=True, slots=True)
class ClaimedOutboxV1:
    id: UUID
    projection_id: UUID
    operation: str
    attempt_count: int


def _instant(value: object) -> datetime:
    if type(value) is not datetime or value.tzinfo is not UTC: raise ValueError("now must be an exact UTC datetime")
    return value


def _identifier(value: object) -> UUID:
    if type(value) is not UUID: raise TypeError("projection identifier must be an exact UUID")
    return value


def _token(value: object, name: str) -> str:
    if type(value) is not str or not value or value != value.strip() or len(value) > 128: raise ValueError(f"{name} must be a bounded canonical token")
    return value


def _checksum(value: object) -> str:
    if type(value) is not str or len(value) != 64 or any(character not in "0123456789abcdef" for character in value): raise ValueError("checksum must be lowercase SHA-256 hexadecimal")
    return value


class FunctionProjectionStateRepository:
    """Fixed calls into migration-owned projection state functions."""

    def __init__(self, state_using: str = "projection_state", source_using: str = "projection_source", *, owner: str | None = None) -> None:
        self.state_using = _token(state_using, "state_using")
        self.source_using = _token(source_using, "source_using")
        self.owner = None if owner is None else _token(owner, "owner")

    def model_read_alias(self, model) -> str:
        if model not in {CollectionGraphProjection, ProjectionChunkReference}: raise ValueError("model is not projection source authority")
        return self.source_using

    def _rows(self, operation: str, parameters: tuple[object, ...]) -> tuple[dict, ...]:
        function = _FUNCTIONS[operation]
        statement = f"SELECT * FROM public.{function}({', '.join(('%s',) * len(parameters))})"
        with connections[self.state_using].cursor() as cursor:
            cursor.execute(statement, parameters)
            if cursor.description is None: return ()
            names = tuple(getattr(item, "name", item[0]) for item in cursor.description)
            return tuple(dict(zip(names, row, strict=True)) for row in cursor.fetchall())

    def _one(self, operation: str, parameters: tuple[object, ...]) -> dict | None:
        function = _FUNCTIONS[operation]
        statement = f"SELECT * FROM public.{function}({', '.join(('%s',) * len(parameters))})"
        with connections[self.state_using].cursor() as cursor:
            cursor.execute(statement, parameters)
            row = cursor.fetchone()
            if row is None: return None
            names = tuple(getattr(item, "name", item[0]) for item in cursor.description)
            return dict(zip(names, row, strict=True))

    def claim(self, *, projection_id: UUID, owner: str, now: datetime, lease_seconds: int):
        if type(lease_seconds) is not int or not 1 <= lease_seconds <= 86_400: raise ValueError("lease_seconds must be an integer in 1..86400")
        row = self._one("claim", (_identifier(projection_id), _token(owner, "owner"), _instant(now), lease_seconds))
        return None if row is None else StateLeaseV1(**row)

    def renew(self, *, projection_id: UUID, owner: str, now: datetime, lease_seconds: int):
        if type(lease_seconds) is not int or not 1 <= lease_seconds <= 86_400: raise ValueError("lease_seconds must be an integer in 1..86400")
        row = self._one("renew", (_identifier(projection_id), _token(owner, "owner"), _instant(now), lease_seconds))
        if row is None: raise RuntimeError("projection lease is not renewable")
        return StateLeaseV1(**row)

    def record_private_mapping(self, *, projection_id: UUID, owner: str, checksum: str, now: datetime) -> None:
        row = self._one("record_mapping", (_identifier(projection_id), _token(owner, "owner"), _checksum(checksum), _instant(now)))
        if row is None: raise RuntimeError("projection lease was lost")

    def fail(self, *, projection_id: UUID, owner: str, failure_code: ProjectionFailureCode, now: datetime) -> None:
        if type(failure_code) is not ProjectionFailureCode: raise TypeError("failure_code must be exact")
        row = self._one("fail", (_identifier(projection_id), _token(owner, "owner"), failure_code.value, _instant(now)))
        if row is None: raise RuntimeError("projection lease was lost")

    def supersede(self, *, projection_id: UUID, now: datetime) -> bool:
        row = self._one("supersede", (_identifier(projection_id), _instant(now)))
        return bool(row and row["changed"])

    def load(self, *, projection_id: UUID, keys: tuple[str, ...] | None = None):
        query = ProjectionChunkReference.objects.using(self.source_using).filter(projection_id=_identifier(projection_id))
        if keys is not None: query = query.filter(projection_chunk_key__in=keys)
        return tuple(query.select_related("chunk").order_by("projection_chunk_key")[:5001])

    def create(self, *, projection_id: UUID, rows: tuple[PrivateProjectionChunkReferenceV1, ...], batch_size: int) -> None:
        del batch_size
        payload = json.dumps(
            [{"projection_chunk_key": row.projection_chunk_key, "integer_chunk_pk": row.integer_chunk_pk, "document_uuid": row.document_uuid, "chunk_number": row.chunk_number} for row in rows],
            sort_keys=True,
            separators=(",", ":"),
        )
        self._one("store_chunks", (_identifier(projection_id), _token(self.owner, "owner"), payload, timezone.now()))

    def fence(self, *, projection_id: UUID, checksum: str, row_count: int) -> None:
        if type(row_count) is not int or not 0 <= row_count <= 5_000: raise ValueError("row_count must be an integer in 0..5000")
        row = self._one("fence_chunks", (_identifier(projection_id), _token(self.owner, "owner"), _checksum(checksum), row_count, timezone.now()))
        if row is None or not row["fenced"]: raise ValueError("projection private mapping fence was lost")

    def claim_outbox(self, *, limit: int, now: datetime) -> tuple[ClaimedOutboxV1, ...]:
        if type(limit) is not int or not 1 <= limit <= 5_000: raise ValueError("limit must be an integer in 1..5000")
        return tuple(ClaimedOutboxV1(**row) for row in self._rows("claim_outbox", (limit, _instant(now))))

    def complete_outbox(self, *, outbox_id: UUID, now: datetime) -> bool:
        row = self._one("complete_outbox", (_identifier(outbox_id), _instant(now)))
        return bool(row and row["changed"])

    def fail_outbox(self, *, outbox_id: UUID, now: datetime, attempt_count: int) -> bool:
        delay = min(300, 2 ** min(attempt_count, 8))
        row = self._one("fail_outbox", (_identifier(outbox_id), "broker_publish_failed", _instant(now) + timedelta(seconds=delay)))
        return bool(row and row["changed"])

    def ready(self, *, projection_id: UUID, owner: str, validation, expected_generation_key: str, expected_graph_checksum: str, expected_private_mapping_checksum: str, now: datetime, versions: tuple[str, str, str]):
        identifier = _identifier(projection_id)
        row = CollectionGraphProjection.objects.using(self.source_using).values("generation_key", "collection_id", "artifact_id", "membership_epoch", "membership_checksum").get(pk=identifier)
        counts = validation.counts
        result = self._one("ready", (
            identifier, row["collection_id"], row["artifact_id"], row["generation_key"], _token(owner, "owner"), *versions,
            row["membership_epoch"], row["membership_checksum"], _checksum(expected_private_mapping_checksum), _checksum(expected_graph_checksum),
            _checksum(validation.validation_checksum), counts.entity_count, counts.relation_semantics_count, counts.relation_count,
            counts.evidence_count, counts.entity_mention_count, counts.chunk_count, bool(validation.valid), _instant(now),
        ))
        if validation.generation_key != _checksum(expected_generation_key): raise ValueError("projection generation validation is stale")
        if result is None: return StateReadyOutcomeV1(identifier, False, "superseded", "source_changed")
        return StateReadyOutcomeV1(identifier, result["published"], result["state"], result["failure_code"])

    def replay(self, *, projection_id: UUID | None, collection_id: int, artifact_id: int, versions: tuple[str, str, str], now: datetime) -> UUID:
        row = self._one("replay", (projection_id, uuid4(), collection_id, artifact_id, *versions, _instant(now)))
        if row is None: raise RuntimeError("projection replay source is stale")
        return row["projection_id"]


__all__ = ["FunctionProjectionStateRepository"]
