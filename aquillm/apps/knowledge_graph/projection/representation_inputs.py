"""Active representation enumeration and captured graph-build input fences."""

from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.services.representation_store import (
    active_chunk_scope,
    representation_snapshot_digest,
    validate_representation_snapshot,
)

from .representation_binding import (
    freeze_document_ids,
    snapshot_payload,
    validate_record_binding,
)


def capture_build_snapshot(document_ids, *, using="default"):
    import os

    from lib.knowledge_graph.retrieval_config import load_hybrid_retrieval_settings

    settings = load_hybrid_retrieval_settings(
        {
            "KG_GRAPH_REMEDIATION_MODE": os.environ.get(
                "KG_GRAPH_REMEDIATION_MODE", "legacy"
            )
        }
    )
    if settings.graph_remediation_mode != "v2":
        from apps.documents.models import DocumentRepresentationState
        if DocumentRepresentationState.objects.using(using).filter(document_id__in=document_ids).exists():
            raise SourceScopeUnavailable("source_scope_unavailable")
        return None
    return freeze_document_ids(document_ids, using=using)


def binding_values(snapshot):
    if snapshot is None:
        return {}
    return {
        "representation_snapshot": snapshot_payload(snapshot),
        "representation_digest": representation_snapshot_digest(snapshot),
    }


def binding_digest(snapshot):
    return None if snapshot is None else representation_snapshot_digest(snapshot)


def active_build_chunks(queryset, document_ids, *, snapshot=None):
    snapshot = (
        capture_build_snapshot(document_ids, using=queryset.db)
        if snapshot is None
        else snapshot
    )
    return (
        queryset
        if snapshot is None
        else active_chunk_scope(queryset, snapshot=snapshot)
    )


def validate_artifact_selection(artifact, selection, *, using):
    snapshot = validate_record_binding(artifact)
    if snapshot is None:
        if (
            selection.generation_key is not None
            or selection.source_revision_sha256 is not None
        ):
            raise SourceScopeUnavailable("source_scope_unavailable")
    elif snapshot.selections != (selection,):
        raise SourceScopeUnavailable("source_scope_unavailable")
    if (
        str(selection.document.id) != artifact.scope_id
        or selection.document.source_hash != artifact.source_hash
    ):
        raise SourceScopeUnavailable("source_scope_unavailable")


def projection_input_snapshot(artifact_id, *, using):
    from apps.knowledge_graph.models import CollectionArtifactInput, GraphArtifact

    artifact = GraphArtifact.objects.using(using).get(pk=artifact_id)
    inputs = tuple(
        CollectionArtifactInput.objects.using(using)
        .filter(artifact_id=artifact_id)
        .select_related("document_artifact")
        .order_by("document_id")
    )
    snapshot = validate_record_binding(artifact)
    if snapshot is None:
        snapshot = freeze_document_ids(
            tuple(i.document_id for i in inputs), using=using
        )
    validate_representation_snapshot(snapshot, using=using)
    selections = {s.document.id: s for s in snapshot.selections}
    if set(selections) != {i.document_id for i in inputs} or len(inputs) != len(
        selections
    ):
        raise SourceScopeUnavailable("source_scope_unavailable")
    for item in inputs:
        if item.collection_id != artifact.collection_scope_id:
            raise SourceScopeUnavailable("source_scope_unavailable")
        validate_artifact_selection(
            item.document_artifact, selections[item.document_id], using=using
        )
    return snapshot


def validate_projection_current(projection, *, using):
    snapshot = validate_record_binding(projection)
    if snapshot is None:
        return  # Exact old path; V2 materialization separately proves legacy-null.
    validate_representation_snapshot(snapshot, using=using)
    if (
        representation_snapshot_digest(
            projection_input_snapshot(projection.artifact_id, using=using)
        )
        != projection.representation_digest
    ):
        raise SourceScopeUnavailable("source_scope_unavailable")
