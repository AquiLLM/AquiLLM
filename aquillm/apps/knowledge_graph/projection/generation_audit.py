from __future__ import annotations

from dataclasses import dataclass

from django.utils import timezone

from apps.knowledge_graph.models import CollectionGraphProjection

from .identifiers import OpaqueProjectionKey, ProjectionIdentifierDomain
from .records import ProjectionLifecycleState
from .serialization import projection_checksum


@dataclass(frozen=True, slots=True)
class GenerationAuditV1:
    replay_reason: str | None
    collection_key: str | None = None


def _opaque_generation(value: str) -> OpaqueProjectionKey:
    return OpaqueProjectionKey(ProjectionIdentifierDomain.COLLECTION, value)


def _projection_page(*, after_id, page_size: int, collection_id: int | None = None):
    query = CollectionGraphProjection.objects.order_by("id")
    if collection_id is not None:
        query = query.filter(collection_pk_snapshot=collection_id)
    if after_id is not None:
        query = query.filter(id__gt=after_id)
    return tuple(query[:page_size])


def _row_matches_bundle(row, bundle, checksum: str) -> bool:
    marker = bundle.generation
    counts = bundle.counts
    return (
        (
            row.schema_version,
            row.projection_version,
            row.identifier_key_version,
        )
        == (
            marker.schema_version,
            marker.projection_version,
            marker.identifier_key_version,
        )
        and row.membership_epoch == marker.membership_epoch
        and row.membership_checksum == marker.membership_checksum
        and row.graph_checksum == checksum
        and row.snapshot_checksum == checksum
        and (
            row.entity_count,
            row.relation_count,
            row.evidence_count,
            row.chunk_count,
        )
        == (
            counts.entity_count,
            counts.relation_count,
            counts.evidence_count,
            counts.chunk_count,
        )
    )


def _manifest_matches(row, bundle, manifest, checksum: str) -> bool:
    marker = bundle.generation
    return (
        manifest.generation_key == marker.generation_key
        and manifest.schema_version == marker.schema_version
        and manifest.projection_version == marker.projection_version
        and manifest.identifier_key_version == marker.identifier_key_version
        and manifest.graph_checksum == checksum
        and manifest.snapshot_checksum == checksum
        and manifest.private_mapping_checksum == row.private_mapping_checksum
        and manifest.counts == bundle.counts
        and manifest.state is ProjectionLifecycleState.READY
    )


def audit_projection_generation(*, row, postgres, graph, settings) -> GenerationAuditV1:
    if row is None:
        return GenerationAuditV1("missing_authority")
    if row.state == "pending":
        return GenerationAuditV1("pending")
    if row.state == "failed":
        return GenerationAuditV1("failed")
    if row.state == "building":
        expired = row.lease_expires_at is None or row.lease_expires_at <= timezone.now()
        return GenerationAuditV1("expired_lease" if expired else None)
    if row.state != "ready":
        return GenerationAuditV1(None)
    bundle = postgres.load_projection_bundle(
        projection_id=row.id,
        batch_size=settings.projection_batch_size,
    )
    checksum = projection_checksum(bundle)
    try:
        manifest = graph.read_generation_manifest(
            generation_key=_opaque_generation(bundle.generation.generation_key),
            timeout_seconds=settings.graph_overall_timeout_ms / 1_000.0,
        )
    except ValueError:
        return GenerationAuditV1("missing_generation", bundle.generation.collection_key)
    if not _row_matches_bundle(row, bundle, checksum):
        return GenerationAuditV1("authority_drift", bundle.generation.collection_key)
    if not _manifest_matches(row, bundle, manifest, checksum):
        return GenerationAuditV1("checksum_drift", bundle.generation.collection_key)
    return GenerationAuditV1(None, bundle.generation.collection_key)


def _authoritative_generation_keys(
    *, postgres, settings, collection_id: int | None
) -> frozenset[str]:
    after_id = None
    values: set[str] = set()
    while True:
        page = _projection_page(
            after_id=after_id,
            page_size=settings.projection_batch_size,
            collection_id=collection_id,
        )
        if not page:
            break
        for row in page:
            bundle = postgres.load_projection_bundle(
                projection_id=row.id,
                batch_size=settings.projection_batch_size,
            )
            values.add(bundle.generation.generation_key)
        after_id = page[-1].id
        if len(page) < settings.projection_batch_size:
            break
    return frozenset(values)


def orphan_generation_keys(
    *,
    postgres,
    graph,
    settings,
    limit: int,
    collection_id: int | None = None,
    collection_key: OpaqueProjectionKey | None = None,
) -> tuple[OpaqueProjectionKey, ...]:
    authoritative = _authoritative_generation_keys(
        postgres=postgres,
        settings=settings,
        collection_id=collection_id,
    )
    after = None
    orphaned: list[OpaqueProjectionKey] = []
    while len(orphaned) < limit:
        page = graph.list_generations(
            collection_key=collection_key,
            after_generation_key=after,
            limit=min(settings.projection_batch_size, limit),
            timeout_seconds=settings.graph_overall_timeout_ms / 1_000.0,
        )
        if not page:
            break
        for manifest in page:
            if manifest.generation_key not in authoritative:
                orphaned.append(_opaque_generation(manifest.generation_key))
                if len(orphaned) == limit:
                    break
        after = _opaque_generation(page[-1].generation_key)
        if len(page) < min(settings.projection_batch_size, limit):
            break
    return tuple(orphaned)


__all__ = [
    "GenerationAuditV1",
    "audit_projection_generation",
    "orphan_generation_keys",
]
