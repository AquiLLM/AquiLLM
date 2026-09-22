"""Current collection graph activity and rebuild status."""

from __future__ import annotations

from django.utils import timezone

from apps.collections.services.graph_progress import selected_graph_ontology_identity
from apps.documents.models import DESCENDED_FROM_DOCUMENT
from apps.knowledge_graph.models import (
    GraphArtifact,
    GraphBuildRun,
    GraphRebuildRequest,
)
from apps.knowledge_graph.models.inputs import (
    collection_input_source_signature,
    collection_manifest_source_hash,
    document_membership_signature,
)
from apps.knowledge_graph.resolution.collection import MAX_COLLECTION_DOCUMENT_INPUTS


def _iso(value):
    return value.isoformat() if value is not None else None


def _active_artifact(collection):
    return (
        GraphArtifact.objects.filter(
            collection_scope=collection,
            scope_type=GraphArtifact.ScopeType.COLLECTION,
            status=GraphArtifact.Status.ACTIVE,
            evaluation_only=False,
        )
        .order_by("-activated_at", "-pk")
        .first()
    )


def _latest_request(collection):
    return (
        GraphRebuildRequest.objects.filter(
            scope_type=GraphRebuildRequest.ScopeType.COLLECTION,
            scope_id=str(collection.pk),
            evaluation_only=False,
        )
        .order_by("-created_at", "-pk")
        .first()
    )


def _current_collection_attempt(collection, progress):
    """Observe activity only; this never grants graph or retrieval readiness."""
    if not 0 < progress["total"] <= MAX_COLLECTION_DOCUMENT_INPUTS:
        return None
    if progress["active"] != progress["total"]:
        return None
    run = (
        GraphBuildRun.objects.filter(
            build_kind="collection",
            scope_type="collection",
            scope_id=str(collection.pk),
            evaluation_only=False,
            orchestration_version=1,
            artifact__collection_scope=collection,
            artifact__scope_type="collection",
            artifact__evaluation_only=False,
            artifact__orchestration_version=1,
        )
        .select_related("artifact")
        .defer(
            "stage_marker",
            "stats",
            "metadata",
            "error_message",
            "error_metadata",
            "artifact__metadata",
        )
        .order_by("-build_generation", "-attempt", "-pk")
        .first()
    )
    if run is None:
        return None
    artifact = run.artifact
    live = (
        run.status == "running"
        and artifact.status == "building"
        and bool(run.lease_owner)
        and run.lease_generation > 0
        and run.lease_expires_at is not None
        and run.lease_expires_at > timezone.now()
    )
    failed = run.status == "failed" and artifact.status == "failed"
    if not (live or failed):
        return None
    if any(
        getattr(run, field) != getattr(artifact, field)
        for field in (
            "build_key",
            "build_generation",
            "source_hash",
            "ontology_version",
            "ontology_checksum",
        )
    ):
        return None
    ontology = selected_graph_ontology_identity(collection.pk)
    if ontology is None or (artifact.ontology_version, artifact.ontology_checksum) != (
        ontology["version"],
        ontology["checksum"],
    ):
        return None

    # Recompute the same manifest source address from bounded, content-free
    # current document rows and active document artifact identities. Never load
    # document text, chunk content, embeddings or the build coordinator here.
    documents = []
    for model in DESCENDED_FROM_DOCUMENT:
        remaining = MAX_COLLECTION_DOCUMENT_INPUTS - len(documents)
        documents.extend(
            model.objects.filter(
                collection=collection,
                ingestion_complete=True,
            )
            .only("id", "collection_id", "full_text_hash")
            .order_by("pk")[: remaining + 1]
        )
        if len(documents) > MAX_COLLECTION_DOCUMENT_INPUTS:
            return None
    if len(documents) != progress["total"]:
        return None
    by_id = {str(row.id): row for row in documents}
    if len(by_id) != len(documents):
        return None
    sources = list(
        GraphArtifact.objects.filter(
            scope_type="document",
            scope_id__in=by_id,
            status="active",
            evaluation_only=False,
            ontology_version=ontology["version"],
            ontology_checksum=ontology["checksum"],
        )
        .defer("metadata")
        .order_by("pk")[: len(documents) + 1]
    )
    if len(sources) != len(documents) or {row.scope_id for row in sources} != set(
        by_id
    ):
        return None
    if any(row.source_hash != by_id[row.scope_id].full_text_hash for row in sources):
        return None
    try:
        source_hash = collection_manifest_source_hash(
            collection_input_source_signature(
                collection_id=collection.pk,
                document_id=by_id[row.scope_id].id,
                document_artifact=row,
                membership_signature=document_membership_signature(by_id[row.scope_id]),
            )
            for row in sources
        )
    except ValueError:
        return None
    return run if source_hash == artifact.source_hash else None


def _status(active_artifact, request, progress, collection_attempt=None):
    if active_artifact is not None:
        return {
            "state": "ready",
            "error_code": None,
            "request_id": str(request.pk) if request is not None else None,
            "updated_at": _iso(active_artifact.updated_at),
        }
    if (
        collection_attempt is not None
        and collection_attempt.status == "failed"
        and request is not None
        and request.status
        in {GraphRebuildRequest.Status.QUEUED, GraphRebuildRequest.Status.RUNNING}
        and request.created_at
        > (
            collection_attempt.finished_at
            or collection_attempt.started_at
            or collection_attempt.created_at
        )
    ):
        collection_attempt = None
    if collection_attempt is not None:
        failed = collection_attempt.status == "failed"
        return {
            "state": "failed" if failed else "building",
            "error_code": "collection_build_failed" if failed else None,
            "request_id": str(request.pk) if request is not None else None,
            "updated_at": _iso(
                collection_attempt.finished_at
                if failed
                else collection_attempt.started_at
            ),
        }
    if request is None:
        state = "empty"
        error_code = None
        if progress["failed"]:
            state = "partial" if progress["active"] else "failed"
            error_code = "document_builds_failed"
        elif progress["total"]:
            state = "building"
        return {
            "state": state,
            "error_code": error_code,
            "request_id": None,
            "updated_at": None,
        }
    state = {
        GraphRebuildRequest.Status.QUEUED: "building",
        GraphRebuildRequest.Status.RUNNING: "building",
        GraphRebuildRequest.Status.PARTIAL: "partial",
        GraphRebuildRequest.Status.FAILED: "failed",
        GraphRebuildRequest.Status.SUCCEEDED: "empty",
    }[request.status]
    return {
        "state": state,
        "error_code": request.error_code or None,
        "request_id": str(request.pk),
        "updated_at": _iso(request.updated_at),
    }
