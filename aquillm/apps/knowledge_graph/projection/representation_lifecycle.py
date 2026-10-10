"""Graph fence called before leaf document locks, inside activation's transaction."""

from uuid import UUID

from django.db import DEFAULT_DB_ALIAS, connections, transaction
from django.utils import timezone

from apps.documents.services.representation_contracts import (
    DocumentRefV1,
    SourceScopeUnavailable,
)
from apps.documents.services.representation_store import (
    document_payload,
    freeze_active_representations,
)
from apps.knowledge_graph.graph import invalidation as lifecycle
from apps.knowledge_graph.models import GraphArtifact, GraphRebuildRequest

from .lifecycle import supersede_artifact_projections_locked


def prepare_document_representation_activation(
    reference: DocumentRefV1,
    collection_ids: tuple[int, ...],
    *,
    expected_source_hash: str,
    old_representation_key: UUID | None,
    new_representation_key: UUID,
    using: str,
) -> tuple[int, ...]:
    """Retain artifacts/chunks, invalidate serving authority and persist rebuild intent.

    I5 must call this before entering its leaf-only document lock context. The
    caller's outer atomic transaction keeps every graph/source lock until the
    pointer update commits. No provider or queue work belongs in this transaction.
    """
    document_payload(reference)
    if using != DEFAULT_DB_ALIAS:
        raise SourceScopeUnavailable("source_scope_unavailable")
    if (
        type(new_representation_key) is not UUID
        or (
            old_representation_key is not None
            and type(old_representation_key) is not UUID
        )
        or new_representation_key == old_representation_key
        or expected_source_hash != reference.source_hash
    ):
        raise SourceScopeUnavailable("source_scope_unavailable")
    if not connections[using].in_atomic_block:
        raise RuntimeError(
            "representation activation requires caller atomic transaction"
        )
    if type(collection_ids) is not tuple or collection_ids != lifecycle._collection_ids(
        collection_ids
    ):
        raise ValueError("collection_ids must be exact sorted unique IDs")
    document = lifecycle.DocumentLifecycleRef(
        reference.model_label, reference.pkid, reference.id
    )
    lock_set = lifecycle._initial_document_lock_set(
        document,
        collection_ids,
        using=using,
        active_or_building_only=True,
    )
    with transaction.atomic(using=using):
        collection_locks, document_locks, row = lifecycle._lock_document_graph_phases(
            document,
            lock_set,
            using=using,
        )
        if row is None or row.collection_id != reference.collection_id:
            raise SourceScopeUnavailable("source_scope_unavailable")
        affected = lifecycle._locked_document_scope_set(
            document,
            row,
            collection_ids,
            using=using,
            active_or_building_only=True,
        )
        lifecycle._raise_if_lock_set_expanded(lock_set, affected)
        snapshot = freeze_active_representations((reference,), using=using)
        if snapshot.selections[0].generation_key != old_representation_key:
            raise SourceScopeUnavailable("source_scope_unavailable")
        collection_artifacts, collection_runs = lifecycle._load_graph_rows(
            collection_locks, using=using
        )
        document_artifacts, document_runs = lifecycle._load_graph_rows(
            document_locks, using=using
        )
        artifacts = tuple(
            a
            for a in (*collection_artifacts, *document_artifacts)
            if a.status in ("active", "building")
        )
        ids = tuple(sorted(a.pk for a in artifacts))
        lifecycle._terminalize_runs(
            lifecycle._runs_for_artifacts((*collection_runs, *document_runs), ids),
            reason="representation_changed",
            using=using,
        )
        supersede_artifact_projections_locked(
            artifact_ids=ids, now=timezone.now(), using=using
        )
        for artifact in artifacts:
            artifact.status = GraphArtifact.Status.SUPERSEDED
            artifact.superseded_at = timezone.now()
            artifact.save(
                using=using, update_fields=["status", "superseded_at", "updated_at"]
            )
        # Existing durable dispatcher owns retries. Publication only follows commit.
        request = GraphRebuildRequest.objects.using(using).create(
            scope_type="document",
            scope_id=str(reference.id),
            requested_documents=[
                {
                    "document_id": str(reference.id),
                    "document_pkid": reference.pkid,
                    "model_label": reference.model_label,
                    "collection_id": reference.collection_id,
                    "source_hash": reference.source_hash,
                    "representation_binding": {
                        "version": "graph-document-representation-v1",
                        "generation_key": str(new_representation_key),
                    },
                }
            ],
            document_count=1,
            collection_count=0,
            status="running",
            document_publication_state="pending",
            collection_publication_state="not_applicable",
            started_at=timezone.now(),
        )

        def publish():
            from apps.knowledge_graph.services.builds import resume_rebuild_request

            resume_rebuild_request(request.pk)

        transaction.on_commit(publish, using=using, robust=True)
        return ids
