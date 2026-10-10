"""Atomic bounded registry publication shared by V1 and fenced V2."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .canonical import CanonicalRebuildResult, CanonicalResolutionResult


def _reconcile_locked_registry(
    *,
    resolution: CanonicalResolutionResult,
    active_artifact_ids: tuple[int, ...],
    source_entities_by_id: Mapping[int, object],
    using: str,
    projected_links=None,
    deadline_check=None,
) -> CanonicalRebuildResult:
    from .canonical import (
        CANONICAL_QUERY_BATCH_SIZE,
        MAX_CANONICAL_REBUILD_ENTITIES,
        MAX_CANONICAL_REBUILD_MEMBERSHIPS,
        CanonicalOutcome,
        CanonicalRebuildResult,
        _batches,
        _bounded_extend,
        _bounded_locked_rows,
        project_canonical_link_decisions,
    )

    check = deadline_check or (lambda: None)
    check()
    from django.db import models as django_models

    from apps.knowledge_graph.models import CanonicalEntity, CanonicalEntityLink

    if projected_links is None:
        projected_links = project_canonical_link_decisions(resolution)
    component_by_identity = {
        component.identity_key: component for component in resolution.components
    }
    if len(component_by_identity) != len(resolution.components):
        raise RuntimeError("canonical component identity is not unique")
    desired_identity_keys = tuple(
        sorted({row.target_identity_key for row in projected_links})
    )
    if any(key not in component_by_identity for key in desired_identity_keys):
        raise RuntimeError("projected canonical target escaped the resolution")
    projected_source_ids = {row.source_entity_id for row in projected_links}
    if not projected_source_ids.issubset(source_entities_by_id):
        raise RuntimeError("projected canonical source escaped the locked entity set")

    registry_rows: list[object] = list(
        _bounded_locked_rows(
            CanonicalEntity.objects.using(using)
            .select_for_update()
            .filter(resolver_version=resolution.resolver_version)
            .exclude(status=CanonicalEntity.Status.SUPERSEDED)
            .order_by("pk")
            .iterator(chunk_size=1_000),
            maximum=MAX_CANONICAL_REBUILD_ENTITIES,
            label="live canonical registry rows",
        )
    )
    for identity_batch in _batches(desired_identity_keys):
        check()
        _bounded_extend(
            registry_rows,
            CanonicalEntity.objects.using(using)
            .select_for_update()
            .filter(
                resolver_version=resolution.resolver_version,
                identity_key__in=identity_batch,
                status=CanonicalEntity.Status.SUPERSEDED,
            )
            .order_by("pk")
            .iterator(chunk_size=1_000),
            maximum=MAX_CANONICAL_REBUILD_ENTITIES * 2,
            label="desired historical canonical registry rows",
        )
    link_rows = _bounded_locked_rows(
        CanonicalEntityLink.objects.using(using)
        .select_for_update(of=("self",))
        .annotate(
            source_collection_id=django_models.F("collection_entity__collection_id")
        )
        .filter(resolver_version=resolution.resolver_version)
        .exclude(status=CanonicalEntityLink.Status.SUPERSEDED)
        .order_by("pk")
        .iterator(chunk_size=1_000),
        maximum=MAX_CANONICAL_REBUILD_MEMBERSHIPS,
        label="live canonical audit links",
    )
    registry_by_identity: dict[str, object] = {}
    for row in registry_rows:
        check()
        if row.identity_key in registry_by_identity:
            raise RuntimeError("canonical identity key is not unique")
        if row.status in {
            CanonicalEntity.Status.SUPPRESSED,
            CanonicalEntity.Status.REJECTED,
        }:
            raise RuntimeError("canonical registry contains a non-lifecycle status")
        registry_by_identity[row.identity_key] = row

    created_entities: list[object] = []
    canonical_by_identity: dict[str, object] = {}
    pending_entities: list[object] = []
    for identity_key in desired_identity_keys:
        check()
        component = component_by_identity[identity_key]
        canonical = registry_by_identity.get(identity_key)
        if canonical is not None:
            if (
                canonical.entity_type != component.entity_type
                or canonical.version_signature != component.version_signature
            ):
                raise RuntimeError(
                    "canonical registry identity conflicts with component type/version"
                )
        else:
            canonical = CanonicalEntity(
                identity_key=component.identity_key,
                resolver_version=resolution.resolver_version,
                label=component.label,
                normalized_label=component.normalized_label,
                entity_type=component.entity_type,
                version_signature=component.version_signature,
                status=CanonicalEntity.Status.ACTIVE,
                embedding=None,
                metadata={
                    "identity_method": component.method,
                    "registry_version": resolution.resolver_version,
                },
            )
            if canonical._raw_validation_errors():
                raise RuntimeError("generated canonical identity failed raw validation")
            canonical.clean()
            pending_entities.append(canonical)
        canonical_by_identity[identity_key] = canonical

    if pending_entities:
        django_models.QuerySet.bulk_create(
            CanonicalEntity.objects.using(using),
            pending_entities,
            batch_size=CANONICAL_QUERY_BATCH_SIZE,
        )
        if any(row.pk is None for row in pending_entities):
            raise RuntimeError("canonical bulk insert did not return primary keys")
        created_entities.extend(pending_entities)
        for canonical in pending_entities:
            registry_by_identity[canonical.identity_key] = canonical

    desired_canonical_ids = tuple(
        sorted({row.pk for row in canonical_by_identity.values()})
    )
    reactivate_ids = tuple(
        sorted(
            row.pk
            for row in canonical_by_identity.values()
            if row.status == CanonicalEntity.Status.SUPERSEDED
        )
    )
    if reactivate_ids:
        changed = CanonicalEntity._transition_registry_status_locked(
            reactivate_ids,
            target=CanonicalEntity.Status.ACTIVE,
            using=using,
        )
        if changed != len(reactivate_ids):
            raise RuntimeError("canonical registry reactivation changed concurrently")
        for row in canonical_by_identity.values():
            if row.pk in reactivate_ids:
                row.status = CanonicalEntity.Status.ACTIVE

    desired_links: dict[tuple[int, int], object] = {}
    for projected in projected_links:
        check()
        canonical = canonical_by_identity[projected.target_identity_key]
        source = source_entities_by_id[projected.source_entity_id]
        link = CanonicalEntityLink(
            collection_entity=source,
            canonical_entity=canonical,
            score=projected.score,
            method=projected.method,
            resolver_version=resolution.resolver_version,
            outcome=projected.outcome.value,
            status=projected.status,
            reason=projected.reason,
            metadata=json.loads(projected.metadata_json),
        )
        link.prepare_for_persistence()
        if link._raw_validation_errors():
            raise RuntimeError("generated canonical link failed raw validation")
        link.clean()
        key = (projected.source_entity_id, canonical.pk)
        if key in desired_links:
            raise RuntimeError("projected canonical link key is not unique")
        desired_links[key] = link

    current_by_key: dict[tuple[int, int], list[object]] = defaultdict(list)
    for row in link_rows:
        check()
        current_by_key[(row.collection_entity_id, row.canonical_entity_id)].append(row)
    keep_link_ids: set[int] = set()
    links_to_create: list[object] = []
    for key, desired in sorted(desired_links.items()):
        check()
        exact = tuple(
            row
            for row in current_by_key.get(key, ())
            if row.score == desired.score
            and row.method == desired.method
            and row.outcome == desired.outcome
            and row.status == desired.status
            and row.reason == desired.reason
            and row.metadata == desired.metadata
            and row.decision_checksum == desired.decision_checksum
        )
        if len(exact) > 1:
            raise RuntimeError("duplicate exact canonical audit links")
        if exact:
            keep_link_ids.add(exact[0].pk)
        else:
            links_to_create.append(desired)

    supersede_link_ids = tuple(
        sorted(row.pk for row in link_rows if row.pk not in keep_link_ids)
    )
    if supersede_link_ids:
        changed = CanonicalEntityLink._supersede_locked(supersede_link_ids, using=using)
        if changed != len(supersede_link_ids):
            raise RuntimeError("canonical link supersession changed concurrently")
    if links_to_create:
        django_models.QuerySet.bulk_create(
            CanonicalEntityLink.objects.using(using),
            links_to_create,
            batch_size=CANONICAL_QUERY_BATCH_SIZE,
        )

    supersede_entity_ids = tuple(
        sorted(
            row.pk
            for row in registry_rows
            if row.status == CanonicalEntity.Status.ACTIVE
            and row.pk not in desired_canonical_ids
        )
    )
    if supersede_entity_ids:
        changed = CanonicalEntity._transition_registry_status_locked(
            supersede_entity_ids,
            target=CanonicalEntity.Status.SUPERSEDED,
            using=using,
        )
        if changed != len(supersede_entity_ids):
            raise RuntimeError("canonical registry supersession changed concurrently")
    superseded_link_ids = set(supersede_link_ids)
    changed_collection_ids = tuple(
        sorted(
            {
                source_entities_by_id[row.collection_entity_id].collection_id
                for row in links_to_create
                if row.outcome == CanonicalEntityLink.Outcome.AUTOMATIC
            }
            | {
                # Historical sources can be absent from the current snapshot.
                row.source_collection_id
                for row in link_rows
                if row.pk in superseded_link_ids
                and row.outcome == CanonicalEntityLink.Outcome.AUTOMATIC
            }
        )
    )
    if changed_collection_ids:
        from apps.knowledge_graph.projection.runtime import (
            enqueue_automatic_membership_projections,
        )

        enqueue_automatic_membership_projections(changed_collection_ids, using=using)
    return CanonicalRebuildResult(
        resolver_version=resolution.resolver_version,
        resolution_checksum=resolution.checksum,
        active_artifact_ids=active_artifact_ids,
        canonical_entity_ids=desired_canonical_ids,
        active_link_count=sum(
            row.outcome is CanonicalOutcome.AUTOMATIC for row in projected_links
        ),
        created_entity_count=len(created_entities),
        created_link_count=len(links_to_create),
        superseded_entity_count=len(supersede_entity_ids),
        superseded_link_count=len(supersede_link_ids),
    )
