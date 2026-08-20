from __future__ import annotations

import json
from collections.abc import Sequence
from hashlib import sha256

from django.conf import settings
from django.db import connections

from apps.collections.models import Collection
from apps.knowledge_graph.models import (
    CanonicalEntityLink,
    CollectionEntity,
    CollectionGraphMembershipState,
    GraphArtifact,
)

from .identifiers import ProjectionIdentifierCodec, ProjectionIdentifierDomain
from .records import AutomaticCanonicalMembershipV1
from .serialization import projection_checksum

_MAX_BATCH = 5_000
MEMBERSHIP_REGISTRY_GENERATION = "membership-registry-v1"


def _batch_size(value: object) -> int:
    if type(value) is not int or not 1 <= value <= _MAX_BATCH:
        raise ValueError("batch_size must be an integer in 1..5000")
    return value


def _collection_ids(value: object) -> tuple[int, ...]:
    if type(value) is not tuple or any(
        type(item) is not int or item < 1 for item in value
    ):
        raise ValueError("collection_ids must contain positive integers")
    if not value or value != tuple(sorted(set(value))):
        raise ValueError("collection_ids must be nonempty, sorted, and unique")
    return value


def _opaque_membership_key(codec: ProjectionIdentifierCodec, primary_key: int) -> str:
    return codec.encode(
        ProjectionIdentifierDomain.AUTOMATIC_CANONICAL_IDENTITY,
        source=primary_key,
    ).value


def _opaque_entity_key(
    codec: ProjectionIdentifierCodec, generation: object, primary_key: int
) -> str:
    return codec.encode(
        ProjectionIdentifierDomain.ENTITY,
        generation=generation,  # type: ignore[arg-type]
        source=primary_key,
    ).value


def _configured_codec() -> ProjectionIdentifierCodec:
    from .identifiers import HmacSha256ProjectionIdentifierCodec

    secret = getattr(settings, "KG_PROJECTION_IDENTIFIER_HMAC_KEY", "")
    version = getattr(settings, "KG_PROJECTION_IDENTIFIER_KEY_VERSION", "")
    if type(secret) is not str or not secret or type(version) is not str or not version:
        raise RuntimeError("projection identifier codec is not configured")
    return HmacSha256ProjectionIdentifierCodec(
        secret.encode("utf-8"), key_version=version
    )


def null_membership_decision_checksum(
    entity_key: str, resolver_version: str, resolution_checksum: str
) -> str:
    payload = {
        "automatic_membership_key": None,
        "entity_key": entity_key,
        "resolution_config_checksum": resolution_checksum,
        "resolver_version": resolver_version,
    }
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _load_membership_source_rows(
    *, collection_ids: tuple[int, ...], using: str, batch_size: int
) -> tuple[tuple[int, str, str, int | None, str | None], ...]:
    entities = (
        CollectionEntity.objects.using(using)
        .filter(
            collection_id__in=collection_ids,
            status="active",
            artifact__status="active",
            artifact__scope_type="collection",
            artifact__evaluation_only=False,
        )
        .order_by("pk")
        .values_list(
            "pk",
            "artifact__resolver_version",
            "artifact__resolution_config_checksum",
        )
    )
    entity_rows = tuple(entities.iterator(chunk_size=batch_size))
    entity_ids = tuple(row[0] for row in entity_rows)
    links = (
        CanonicalEntityLink.objects.using(using)
        .filter(
            collection_entity_id__in=entity_ids,
            outcome="automatic",
            status="active",
            canonical_entity__status="active",
        )
        .order_by("collection_entity_id", "pk")
        .values_list("collection_entity_id", "canonical_entity_id", "decision_checksum")
    )
    by_entity: dict[int, tuple[int, str]] = {}
    for entity_id, canonical_id, decision_checksum in links.iterator(
        chunk_size=batch_size
    ):
        if entity_id in by_entity:
            raise ValueError("automatic membership is not unique")
        by_entity[entity_id] = (canonical_id, decision_checksum)
    return tuple(
        (entity_id, resolver, checksum, *by_entity.get(entity_id, (None, None)))
        for entity_id, resolver, checksum in entity_rows
    )


def load_automatic_membership_assignments(
    *,
    collection_ids: tuple[int, ...],
    using: str,
    batch_size: int,
    codec: ProjectionIdentifierCodec | None = None,
    generation: object = MEMBERSHIP_REGISTRY_GENERATION,
) -> tuple[AutomaticCanonicalMembershipV1, ...]:
    ids = _collection_ids(collection_ids)
    size = _batch_size(batch_size)
    if type(using) is not str or not using:
        raise ValueError("using must be a nonempty database alias")
    encoder = codec if codec is not None else _configured_codec()
    assignments = []
    for (
        entity_id,
        resolver,
        resolution_checksum,
        canonical_id,
        decision,
    ) in _load_membership_source_rows(
        collection_ids=ids,
        using=using,
        batch_size=size,
    ):
        entity_key = _opaque_entity_key(encoder, generation, entity_id)
        membership_key = (
            None
            if canonical_id is None
            else _opaque_membership_key(encoder, canonical_id)
        )
        assignments.append(
            AutomaticCanonicalMembershipV1(
                entity_key=entity_key,
                automatic_membership_key=membership_key,
                decision_checksum=(
                    decision
                    if decision is not None
                    else null_membership_decision_checksum(
                        entity_key, resolver, resolution_checksum
                    )
                ),
                resolver_version=resolver,
                resolution_config_checksum=resolution_checksum,
            )
        )
    return tuple(sorted(assignments, key=lambda row: row.entity_key))


def membership_decision_checksum(
    assignments: Sequence[AutomaticCanonicalMembershipV1],
) -> str:
    if isinstance(assignments, (str, bytes)):
        raise TypeError("assignments must be projection membership rows")
    rows = tuple(assignments)
    if any(type(row) is not AutomaticCanonicalMembershipV1 for row in rows):
        raise TypeError("assignments must contain exact membership rows")
    rows = tuple(sorted(rows, key=lambda row: row.entity_key))
    if not rows:
        return sha256(b"[]").hexdigest()
    return projection_checksum(rows)


def advance_membership_state_locked(
    *, collection_id: int, using: str, expected_artifact_id: int | None
) -> CollectionGraphMembershipState:
    if type(collection_id) is not int or collection_id < 1:
        raise ValueError("collection_id must be a positive integer")
    if expected_artifact_id is not None and (
        type(expected_artifact_id) is not int or expected_artifact_id < 1
    ):
        raise ValueError("expected_artifact_id must be a positive integer or null")
    if not connections[using].in_atomic_block:
        raise RuntimeError("membership advancement requires an atomic transaction")
    Collection.objects.using(using).select_for_update().get(pk=collection_id)
    artifact = (
        GraphArtifact.objects.using(using)
        .filter(
            scope_type="collection",
            scope_id=str(collection_id),
            status="active",
            evaluation_only=False,
        )
        .only("pk", "resolver_version", "resolution_config_checksum")
        .first()
    )
    artifact_id = None if artifact is None else artifact.pk
    if artifact_id != expected_artifact_id:
        raise ValueError("active artifact changed while membership state was locked")
    assignments = load_automatic_membership_assignments(
        collection_ids=(collection_id,), using=using, batch_size=_MAX_BATCH
    )
    checksum = membership_decision_checksum(assignments)
    resolver = "unresolved-v1" if artifact is None else artifact.resolver_version
    config = (
        sha256(b"unresolved-v1").hexdigest()
        if artifact is None
        else artifact.resolution_config_checksum
    )
    state, created = (
        CollectionGraphMembershipState.objects.using(using)
        .select_for_update()
        .get_or_create(
            collection_id=collection_id,
            defaults={
                "active_artifact_id": artifact_id,
                "membership_checksum": checksum,
                "resolver_version": resolver,
                "resolution_config_checksum": config,
            },
        )
    )
    current = (
        state.active_artifact_id,
        state.membership_checksum,
        state.resolver_version,
        state.resolution_config_checksum,
    )
    updated = (artifact_id, checksum, resolver, config)
    if not created and current != updated:
        state.registry_epoch += 1
        state.active_artifact_id = artifact_id
        state.membership_checksum = checksum
        state.resolver_version = resolver
        state.resolution_config_checksum = config
        state.save(using=using)
    return state
