"""Unchanged legacy locking path retained for default mode and rollback."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .canonical import CanonicalRebuildResult


def _rebuild_canonical_snapshot(
    snapshot: tuple[tuple[int, int], ...],
    *,
    using: str,
    identity_policy: str = "legacy",
) -> CanonicalRebuildResult:
    from django.db import connections, transaction
    from django.db.models import F

    from apps.collections.models import Collection
    from apps.knowledge_graph.graph.assembly import _lock_collection_scope
    from apps.knowledge_graph.models import CollectionEntity, GraphArtifact

    from .canonical import (
        MAX_CANONICAL_REBUILD_ENTITIES,
        _active_collection_artifact_snapshot,
        _bounded_locked_rows,
        _CanonicalSnapshotChanged,
        _derive_locked_embedding_candidates,
        _load_locked_canonical_inputs,
        _lock_canonical_registry,
        _reconcile_locked_registry,
        resolve_canonical_entities,
    )

    collection_ids = tuple(sorted(collection_id for _, collection_id in snapshot))
    artifact_ids = tuple(sorted(artifact_id for artifact_id, _ in snapshot))
    with transaction.atomic(using=using):
        # Task 13 lock protocol: take the complete sorted C scope/row/artifact
        # prefix before the canonical advisory or any canonical/link row.
        with connections[using].cursor() as cursor:
            for collection_id in collection_ids:
                _lock_collection_scope(cursor, collection_id)
        locked_collections = tuple(
            Collection.objects.using(using)
            .select_for_update()
            .filter(pk__in=collection_ids)
            .only("pk")
            .order_by("pk")
        )
        if tuple(row.pk for row in locked_collections) != collection_ids:
            raise _CanonicalSnapshotChanged("active collection set changed")
        locked_artifacts = tuple(
            GraphArtifact.objects.using(using)
            .select_for_update()
            .filter(
                pk__in=artifact_ids,
                scope_type=GraphArtifact.ScopeType.COLLECTION,
                status=GraphArtifact.Status.ACTIVE,
            )
            .only("pk", "collection_scope_id", "scope_id", "embedding_model_signature",
                  "metadata", "scope_type", "resolver_version", "resolution_config_checksum")
            .order_by("pk")
        )
        if tuple(row.pk for row in locked_artifacts) != artifact_ids:
            raise _CanonicalSnapshotChanged("active collection artifact set changed")
        expected_scope = {
            artifact_id: collection_id for artifact_id, collection_id in snapshot
        }
        if any(
            row.collection_scope_id != expected_scope[row.pk]
            or row.scope_id != str(expected_scope[row.pk])
            for row in locked_artifacts
        ):
            raise _CanonicalSnapshotChanged("active collection artifact scope changed")

        entity_rows = _bounded_locked_rows(
            CollectionEntity.objects.using(using)
            .select_for_update(of=("self",))
            .filter(
                artifact_id__in=artifact_ids,
                artifact__status=GraphArtifact.Status.ACTIVE,
                status=CollectionEntity.Status.ACTIVE,
                collection_id=F("artifact__collection_scope_id"),
            )
            .defer("metadata")
            .order_by("pk")
            .iterator(chunk_size=1_000),
            maximum=MAX_CANONICAL_REBUILD_ENTITIES,
            label="active collection entities",
        )
        from .vocabulary_builds import capture_canonical_artifacts
        vocabulary = capture_canonical_artifacts(locked_artifacts, using=using)
        inputs = _load_locked_canonical_inputs(
            entity_rows=entity_rows,
            active_artifact_ids=artifact_ids,
            using=using,
            vocabulary=vocabulary,
        )
        if (
            _active_collection_artifact_snapshot(
                using=using, identity_policy=identity_policy
            )
            != snapshot
        ):
            raise _CanonicalSnapshotChanged("canonical active snapshot expanded")
        resolution = resolve_canonical_entities(
            inputs,
            identity_policy=identity_policy,
            vocabulary=vocabulary,
            embedding_candidates=_derive_locked_embedding_candidates(
                entity_rows,
                artifact_embedding_signatures={
                    row.pk: row.embedding_model_signature for row in locked_artifacts
                },
            ),
        )

        if capture_canonical_artifacts(locked_artifacts, using=using) != vocabulary:
            raise _CanonicalSnapshotChanged("canonical vocabulary changed")
        _lock_canonical_registry(using=using)
        return _reconcile_locked_registry(
            resolution=resolution,
            active_artifact_ids=artifact_ids,
            source_entities_by_id={row.pk: row for row in entity_rows},
            using=using,
        )


def schedule_canonical_rebuild(*, using: str = "default") -> None:
    """Register one robust post-commit full-registry reconciliation per transaction."""

    from django.db import transaction

    connection = transaction.get_connection(using)
    callback_marker = "_kg_canonical_rebuild_using"
    if any(
        getattr(entry[1], callback_marker, None) == using
        for entry in connection.run_on_commit
    ):
        return

    def reconcile() -> None:
        from .canonical import rebuild_canonical_registry

        rebuild_canonical_registry(using=using)

    setattr(reconcile, callback_marker, using)
    transaction.on_commit(reconcile, using=using, robust=True)
