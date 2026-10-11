"""Two-phase graph-first activation, retaining the immutable predecessor fence."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from uuid import UUID

from django.db import DEFAULT_DB_ALIAS, connections, transaction
from django.utils import timezone

from apps.documents.services.representation_contracts import RepresentationSnapshotV1, SourceScopeUnavailable
from apps.documents.services.representation_store import document_payload, validate_document_reference, validate_representation_snapshot
from apps.knowledge_graph.graph import invalidation as lifecycle
from apps.knowledge_graph.models import GraphArtifact, GraphRebuildRequest
from .lifecycle import supersede_artifact_projections_locked

_active = ContextVar("locked_representation_transition", default=None)


@dataclass(eq=False)
class _LockedRepresentationTransition:
    predecessor: object
    new_key: UUID
    row: object
    artifacts: tuple
    runs: tuple
    connection: object
    atomic: object
    using: str
    finished: bool = False


@contextmanager
def locked_document_representation_transition(predecessor, collection_ids, *, expected_source_hash,
                                               old_representation_key, new_representation_key, using):
    document_payload(predecessor)
    connection = connections[using]
    if (using != DEFAULT_DB_ALIAS or not connection.in_atomic_block or _active.get() is not None or
        type(new_representation_key) is not UUID or new_representation_key == old_representation_key or
        expected_source_hash != predecessor.source_hash or type(collection_ids) is not tuple or
        collection_ids != lifecycle._collection_ids(collection_ids)):
        raise SourceScopeUnavailable("invalid_representation_transition")
    ref = lifecycle.DocumentLifecycleRef(predecessor.model_label, predecessor.pkid, predecessor.id)
    lock_set = lifecycle._initial_document_lock_set(ref, collection_ids, using=using, active_or_building_only=True)
    collection_locks, document_locks, row = lifecycle._lock_document_graph_phases(ref, lock_set, using=using)
    if row is None or row.collection_id != predecessor.collection_id:
        raise SourceScopeUnavailable("source_scope_unavailable")
    affected = lifecycle._locked_document_scope_set(ref, row, collection_ids, using=using, active_or_building_only=True)
    lifecycle._raise_if_lock_set_expanded(lock_set, affected)
    from apps.documents.models import DocumentRepresentationState
    from apps.documents.services.representation_history import validate_locked_state
    validate_document_reference(predecessor, using=using)
    state = DocumentRepresentationState.objects.using(using).select_for_update().get(
        model_label=predecessor.model_label, document_pkid=predecessor.pkid, document_id=predecessor.id)
    validate_locked_state(state, predecessor, using=using)
    if state.active_generation_id != old_representation_key:
        raise SourceScopeUnavailable("source_scope_unavailable")
    ca, cr = lifecycle._load_graph_rows(collection_locks, using=using)
    da, dr = lifecycle._load_graph_rows(document_locks, using=using)
    token = _LockedRepresentationTransition(predecessor, new_representation_key, row,
        tuple(a for a in (*ca, *da) if a.status in ("active", "building")), (*cr, *dr),
        connection, connection.atomic_blocks[-1], using)
    marker = (using, predecessor.model_label, predecessor.pkid, predecessor.id)
    setattr(row, lifecycle._DOCUMENT_ROW_LOCK_CONTEXT, marker)
    context = _active.set(token)
    try:
        yield token
        if not token.finished:
            raise SourceScopeUnavailable("unfinished_representation_transition")
    finally:
        _active.reset(context)
        delattr(row, lifecycle._DOCUMENT_ROW_LOCK_CONTEXT)


def _validate_token(token, using):
    connection = connections[using]
    if (type(token) is not _LockedRepresentationTransition or _active.get() is not token or
        token.finished or token.using != using or token.connection is not connection or
        not connection.in_atomic_block or token.atomic not in connection.atomic_blocks):
        raise SourceScopeUnavailable("invalid_representation_transition")


def finish_document_representation_transition(token, *, target, using):
    _validate_token(token, using)
    before, after = token.predecessor, target.document
    if ((before.model_label, before.pkid, before.id, before.collection_id) !=
        (after.model_label, after.pkid, after.id, after.collection_id) or target.generation_key != token.new_key):
        raise SourceScopeUnavailable("invalid_representation_target")
    validate_representation_snapshot(RepresentationSnapshotV1((target,)), using=using)
    ids = tuple(sorted(a.pk for a in token.artifacts))
    lifecycle._terminalize_runs(lifecycle._runs_for_artifacts(token.runs, ids),
                               reason="representation_changed", using=using)
    supersede_artifact_projections_locked(artifact_ids=ids, now=timezone.now(), using=using)
    for artifact in token.artifacts:
        artifact.status, artifact.superseded_at = GraphArtifact.Status.SUPERSEDED, timezone.now()
        artifact.save(using=using, update_fields=["status", "superseded_at", "updated_at"])
    request = GraphRebuildRequest.objects.using(using).create(
        scope_type="document", scope_id=str(after.id), requested_documents=[{
            "document_id": str(after.id), "document_pkid": after.pkid, "model_label": after.model_label,
            "collection_id": after.collection_id, "source_hash": after.source_hash,
            "representation_binding": {"version": "graph-document-representation-v1", "generation_key": str(token.new_key)},
        }], document_count=1, collection_count=0, status="running", document_publication_state="pending",
        collection_publication_state="not_applicable", started_at=timezone.now(),
    )
    def publish():
        from apps.knowledge_graph.services.builds import resume_rebuild_request
        resume_rebuild_request(request.pk)
    transaction.on_commit(publish, using=using, robust=True)
    token.finished = True
    return ids


def matching_representation_save(instance, *, using):
    """Only the exact locked target row can consume its content lifecycle event."""
    token = _active.get()
    if token is None or instance is not token.row:
        return False
    _validate_token(token, using)
    from apps.documents.models import DocumentRepresentationGeneration
    target = DocumentRepresentationGeneration.objects.using(using).filter(
        pk=token.new_key, status="staged", owner__model_label=instance._meta.label_lower,
        owner__document_pkid=instance.pkid, owner__document_id=instance.id,
        predecessor_source_hash=token.predecessor.source_hash,
        canonical_text=instance.full_text, canonical_sha256=instance.full_text_hash,
        collection_id=instance.collection_id, lease_expires_at__gt=timezone.now(),
    ).exists()
    if not target:
        raise SourceScopeUnavailable("invalid_representation_save")
    return True
