# ruff: noqa: E501, E701, E702, I001
from __future__ import annotations

# fmt: off
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from uuid import UUID
from django.db import connections, transaction
from django.utils import timezone
from apps.collections.models import Collection
from apps.knowledge_graph.models import CollectionGraphMembershipState, CollectionGraphProjection, GraphArtifact, GraphProjectionOutbox
from .memberships import advance_membership_state_locked
from .identifiers import ProjectionIdentifierCodec
from .memgraph_repository import ProjectionValidationV1
from .records import ProjectionFailureCode, ProjectionLeaseV1
from .runtime import load_projection_runtime_settings

_EMPTY_MAPPING_CHECKSUM = sha256(b"[]").hexdigest()
_MAX_ROWS = 5_000
_READY_LOCK_ORDER = ("collection", "active_artifact", "membership_state", "projection")

@dataclass(frozen=True, slots=True)
class ProjectionReadyOutcomeV1:
    projection_id: UUID
    published: bool
    state: str
    failure_code: str | None

def _identifier(value: object) -> UUID:
    if type(value) is not UUID: raise TypeError("projection_id must be an exact UUID")
    return value

def _instant(value: object) -> datetime:
    if type(value) is not datetime or value.tzinfo is not UTC: raise ValueError("now must be an exact UTC datetime")
    return value

def _owner(value: object) -> str:
    if type(value) is not str or not value or value != value.strip() or len(value) > 128: raise ValueError("owner must be a bounded canonical token")
    return value

def _atomic(using: str): return transaction.atomic(using=using)
def _locked_projection(projection_id: UUID, using: str): return CollectionGraphProjection.objects.using(using).select_for_update().get(pk=projection_id)
def _save(row: object, fields: list[str], using: str) -> None: row.save(using=using, update_fields=fields)
def _projection_versions() -> tuple[str, str, str]:
    settings = load_projection_runtime_settings()
    return settings.projection_schema_version, settings.projection_format_version, settings.projection_identifier_key_version
def _checksum(value: object) -> str:
    if type(value) is not str or len(value) != 64 or any(character not in "0123456789abcdef" for character in value): raise ValueError("checksum must be lowercase SHA-256 hexadecimal")
    return value

def _supersede(row: object, now: datetime, using: str) -> None:
    if row.state == CollectionGraphProjection.State.SUPERSEDED: return
    row.state = CollectionGraphProjection.State.SUPERSEDED
    row.lease_owner, row.lease_expires_at, row.failure_code = "", None, ""
    row.superseded_at = now
    _save(row, ["state", "lease_owner", "lease_expires_at", "failure_code", "superseded_at", "updated_at"], using)

def _enqueue_outbox(row: object, operation: str, now: datetime, using: str) -> None:
    entry, created = GraphProjectionOutbox.objects.using(using).get_or_create(projection_id=row.id, operation=operation, defaults={"state": GraphProjectionOutbox.State.PENDING, "next_attempt_at": now})
    if not created:
        entry.state, entry.published_at, entry.next_attempt_at, entry.last_failure_code = GraphProjectionOutbox.State.PENDING, None, now, ""
        entry.save(using=using, update_fields=["state", "published_at", "next_attempt_at", "last_failure_code"])

def enqueue_collection_projection_locked(*, collection_id: int, artifact_id: int, using: str, codec: ProjectionIdentifierCodec) -> CollectionGraphProjection:
    if type(collection_id) is not int or collection_id < 1 or type(artifact_id) is not int or artifact_id < 1: raise ValueError("collection_id and artifact_id must be positive integers")
    with _atomic(using):
        Collection.objects.using(using).select_for_update().get(pk=collection_id)
        GraphArtifact.objects.using(using).select_for_update().get(pk=artifact_id, collection_scope_id=collection_id, scope_type=GraphArtifact.ScopeType.COLLECTION, status=GraphArtifact.Status.ACTIVE, evaluation_only=False)
        membership = advance_membership_state_locked(collection_id=collection_id, using=using, expected_artifact_id=artifact_id, codec=codec)
        rows = tuple(CollectionGraphProjection.objects.using(using).select_for_update().filter(collection_id=collection_id, state__in=("pending", "building", "ready")).order_by("id")[: _MAX_ROWS + 1])
        if len(rows) > _MAX_ROWS: raise RuntimeError("projection fanout exceeds the bounded row limit")
        schema_version, projection_version, identifier_key_version = _projection_versions()
        identity = (artifact_id, membership.registry_epoch, membership.membership_checksum, schema_version, projection_version, identifier_key_version)
        current = next((row for row in rows if (row.artifact_id, row.membership_epoch, row.membership_checksum, row.schema_version, row.projection_version, row.identifier_key_version) == identity), None)
        for row in rows:
            if row is not current:
                instant = timezone.now(); _supersede(row, instant, using); _enqueue_outbox(row, GraphProjectionOutbox.Operation.PRUNE, instant, using)
        if current is None:
            current = CollectionGraphProjection.objects.using(using).create(collection_id=collection_id, collection_pk_snapshot=collection_id, artifact_id=artifact_id, artifact_pk_snapshot=artifact_id, schema_version=schema_version, projection_version=projection_version, identifier_key_version=identifier_key_version, membership_epoch=membership.registry_epoch, membership_checksum=membership.membership_checksum, private_mapping_checksum=_EMPTY_MAPPING_CHECKSUM)
        _enqueue_outbox(current, GraphProjectionOutbox.Operation.PROJECT, timezone.now(), using)
        return current

def enqueue_automatic_membership_changes_locked(*, collection_ids: tuple[int, ...], using: str, codec: ProjectionIdentifierCodec) -> int:
    if type(collection_ids) is not tuple or collection_ids != tuple(sorted(set(collection_ids))) or any(type(value) is not int or value < 1 for value in collection_ids) or len(collection_ids) > _MAX_ROWS: raise ValueError("collection_ids must be a bounded sorted unique tuple")
    if not connections[using].in_atomic_block: raise RuntimeError("membership projection enqueue requires an atomic transaction")
    enqueued = 0
    for collection_id in collection_ids:
        artifact_id = GraphArtifact.objects.using(using).filter(collection_scope_id=collection_id, scope_type="collection", status="active", evaluation_only=False).values_list("pk", flat=True).first()
        if artifact_id is not None:
            enqueue_collection_projection_locked(collection_id=collection_id, artifact_id=artifact_id, using=using, codec=codec); enqueued += 1
    return enqueued

def claim_projection_lease(*, projection_id: UUID, owner: str, now: datetime, lease_seconds: int, using: str) -> ProjectionLeaseV1 | None:
    identifier, lease_owner, instant = _identifier(projection_id), _owner(owner), _instant(now)
    if type(lease_seconds) is not int or not 1 <= lease_seconds <= 86_400: raise ValueError("lease_seconds must be an integer in 1..86400")
    with _atomic(using):
        row = _locked_projection(identifier, using)
        if row.state in {"ready", "superseded"}: return None
        if row.state == "building" and row.lease_expires_at > instant:
            if row.lease_owner != lease_owner: return None
            return ProjectionLeaseV1(str(row.id), lease_owner, row.lease_expires_at, row.attempt_count)
        row.state, row.failure_code = "building", ""; row.attempt_count += 1
        row.lease_owner, row.lease_expires_at = lease_owner, instant + timedelta(seconds=lease_seconds)
        _save(row, ["state", "failure_code", "attempt_count", "lease_owner", "lease_expires_at", "updated_at"], using)
        return ProjectionLeaseV1(str(row.id), lease_owner, row.lease_expires_at, row.attempt_count)

def renew_projection_lease(*, projection_id: UUID, owner: str, now: datetime, lease_seconds: int, using: str) -> ProjectionLeaseV1:
    identifier, lease_owner, instant = _identifier(projection_id), _owner(owner), _instant(now)
    if type(lease_seconds) is not int or not 1 <= lease_seconds <= 86_400: raise ValueError("lease_seconds must be an integer in 1..86400")
    with _atomic(using):
        row = _locked_projection(identifier, using)
        if row.state != "building" or row.lease_owner != lease_owner or row.lease_expires_at <= instant: raise RuntimeError("projection lease is not renewable")
        row.lease_expires_at = instant + timedelta(seconds=lease_seconds); _save(row, ["lease_expires_at", "updated_at"], using)
        return ProjectionLeaseV1(str(row.id), lease_owner, row.lease_expires_at, row.attempt_count)

def record_projection_private_mapping_checksum(*, projection_id: UUID, owner: str, checksum: str, now: datetime, using: str) -> None:
    identifier, lease_owner, instant, value = _identifier(projection_id), _owner(owner), _instant(now), _checksum(checksum)
    with _atomic(using):
        row = _locked_projection(identifier, using)
        if row.state != "building" or row.lease_owner != lease_owner or row.lease_expires_at <= instant: raise RuntimeError("projection lease was lost")
        row.private_mapping_checksum = value
        _save(row, ["private_mapping_checksum", "updated_at"], using)

def mark_projection_failed(*, projection_id: UUID, owner: str, failure_code: ProjectionFailureCode, now: datetime, using: str) -> None:
    identifier, lease_owner, instant = _identifier(projection_id), _owner(owner), _instant(now)
    if type(failure_code) is not ProjectionFailureCode: raise TypeError("failure_code must be exact")
    with _atomic(using):
        row = _locked_projection(identifier, using)
        if row.state != "building" or row.lease_owner != lease_owner or row.lease_expires_at <= instant: raise RuntimeError("projection lease was lost")
        row.state, row.failure_code, row.lease_owner, row.lease_expires_at = "failed", failure_code.value, "", None
        _save(row, ["state", "failure_code", "lease_owner", "lease_expires_at", "updated_at"], using)

def _locked_ready_context(projection_id: UUID, using: str):
    snapshot = CollectionGraphProjection.objects.using(using).only("collection_id", "artifact_id").get(pk=projection_id)
    collection = Collection.objects.using(using).select_for_update().filter(pk=snapshot.collection_id).first()
    artifact = GraphArtifact.objects.using(using).select_for_update().filter(pk=snapshot.artifact_id).first()
    membership = CollectionGraphMembershipState.objects.using(using).select_for_update().filter(collection_id=snapshot.collection_id).first()
    return collection, artifact, membership, _locked_projection(projection_id, using)

def publish_projection_ready_compare_and_set(*, projection_id: UUID, owner: str, validation: ProjectionValidationV1, expected_generation_key: str, expected_graph_checksum: str, expected_private_mapping_checksum: str, now: datetime, using: str) -> ProjectionReadyOutcomeV1:
    identifier, lease_owner, instant = _identifier(projection_id), _owner(owner), _instant(now)
    generation_key, graph_checksum, private_checksum = _checksum(expected_generation_key), _checksum(expected_graph_checksum), _checksum(expected_private_mapping_checksum)
    if type(validation) is not ProjectionValidationV1: raise TypeError("validation must be exact")
    with _atomic(using):
        collection, artifact, membership, row = _locked_ready_context(identifier, using)
        stable = collection is not None and artifact is not None and membership is not None and artifact.pk == row.artifact_id and artifact.status == "active" and getattr(artifact, "evaluation_only", False) is False and membership.active_artifact_id == row.artifact_id and membership.registry_epoch == row.membership_epoch and membership.membership_checksum == row.membership_checksum and membership.resolver_version == artifact.resolver_version and membership.resolution_config_checksum == artifact.resolution_config_checksum and row.collection_id == collection.pk and (row.schema_version, row.projection_version, row.identifier_key_version) == _projection_versions() and row.private_mapping_checksum == private_checksum and row.state == "building" and row.lease_owner == lease_owner and row.lease_expires_at > instant and validation.valid and validation.generation_key == generation_key and validation.validation_checksum == graph_checksum
        if not stable:
            if row.state == "building" and row.lease_owner == lease_owner:
                _supersede(row, instant, using); _enqueue_outbox(row, GraphProjectionOutbox.Operation.PRUNE, instant, using)
            return ProjectionReadyOutcomeV1(identifier, False, row.state, "source_changed")
        counts = validation.counts; row.graph_checksum = row.snapshot_checksum = graph_checksum
        row.entity_count, row.relation_count, row.evidence_count, row.chunk_count = counts.entity_count, counts.relation_count, counts.evidence_count, counts.chunk_count
        row.state, row.ready_at, row.lease_owner, row.lease_expires_at = "ready", instant, "", None
        _save(row, ["graph_checksum", "snapshot_checksum", "entity_count", "relation_count", "evidence_count", "chunk_count", "state", "ready_at", "lease_owner", "lease_expires_at", "updated_at"], using)
        return ProjectionReadyOutcomeV1(identifier, True, "ready", None)

def supersede_projection_locked(*, projection_id: UUID, now: datetime, using: str) -> None:
    with _atomic(using):
        row = _locked_projection(_identifier(projection_id), using); _supersede(row, _instant(now), using); _enqueue_outbox(row, "prune", now, using)

def _projection_rows_for(*, using: str, **filters):
    rows = tuple(CollectionGraphProjection.objects.using(using).select_for_update().filter(**filters).order_by("id")[: _MAX_ROWS + 1])
    if len(rows) > _MAX_ROWS: raise RuntimeError("projection fanout exceeds the bounded row limit")
    return rows

def supersede_artifact_projections_locked(*, artifact_ids: tuple[int, ...], now: datetime, using: str) -> int:
    if type(artifact_ids) is not tuple or artifact_ids != tuple(sorted(set(artifact_ids))): raise ValueError("artifact_ids must be sorted and unique")
    if not connections[using].in_atomic_block: raise RuntimeError("artifact projection supersession requires an atomic transaction")
    changed = 0
    for row in _projection_rows_for(using=using, artifact_id__in=artifact_ids, state__in=("pending", "building", "ready")):
        _supersede(row, _instant(now), using); _enqueue_outbox(row, "prune", now, using); changed += 1
    return changed

def tombstone_collection_projections_locked(*, collection_id: int, now: datetime, using: str) -> int:
    if type(collection_id) is not int or collection_id < 1: raise ValueError("collection_id must be positive")
    if not connections[using].in_atomic_block: raise RuntimeError("collection projection tombstoning requires an atomic transaction")
    changed = 0
    for row in _projection_rows_for(using=using, collection_id=collection_id, state__in=("pending", "building", "ready", "failed")):
        _supersede(row, _instant(now), using); _enqueue_outbox(row, "prune", now, using); changed += 1
    return changed
# fmt: on
