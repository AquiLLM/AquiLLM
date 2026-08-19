"""Artifact/run cleanup fences for synthetic fixture graph occurrences."""

from __future__ import annotations

from uuid import UUID

from django.db.models import F, Q

from .fixture_seed_contract import FixtureSeedError
from .fixture_seed_graph_requests import GraphContext


def _scope_collection(row, context: GraphContext):
    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun

    if row.scope_type == GraphArtifact.ScopeType.COLLECTION:
        try:
            collection_id = int(row.scope_id)
        except (TypeError, ValueError):
            return None
        build_kind = getattr(row, "build_kind", GraphBuildRun.BuildKind.COLLECTION)
        if (
            collection_id not in context.collection_ids
            or row.scope_id != str(collection_id)
            or build_kind != GraphBuildRun.BuildKind.COLLECTION
        ):
            return None
        if (
            hasattr(row, "collection_scope_id")
            and row.collection_scope_id != collection_id
        ):
            return None
        return collection_id
    if row.scope_type != GraphArtifact.ScopeType.DOCUMENT:
        return None
    try:
        document_id = UUID(row.scope_id)
    except (AttributeError, TypeError, ValueError):
        return None
    build_kind = getattr(row, "build_kind", GraphBuildRun.BuildKind.DOCUMENT)
    if (
        str(document_id) != row.scope_id
        or build_kind != GraphBuildRun.BuildKind.DOCUMENT
        or row.source_hash != context.document_hashes.get(document_id)
        or (hasattr(row, "collection_scope_id") and row.collection_scope_id is not None)
    ):
        return None
    return context.document_collections.get(document_id)


def _load_artifacts(context: GraphContext):
    from apps.knowledge_graph.models import GraphArtifact

    expected_request_ids = set(context.expected_requests.values())
    artifacts = tuple(
        GraphArtifact.objects.filter(
            Q(rebuild_request_id__in=expected_request_ids)
            | Q(collection_scope_id__in=context.collection_ids)
            | Q(
                scope_type=GraphArtifact.ScopeType.DOCUMENT,
                scope_id__in=map(str, context.document_collections),
            )
        ).order_by("pk")[:5_001]
    )
    if len(artifacts) > 5_000:
        raise FixtureSeedError(
            "fixture database topology has a foreign graph reference"
        )
    result = {}
    for artifact in artifacts:
        collection_id = _scope_collection(artifact, context)
        if (
            collection_id is None
            or artifact.rebuild_request_id
            != context.expected_requests.get(collection_id)
            or artifact.evaluation_only is not True
            or artifact.status
            in {GraphArtifact.Status.BUILDING, GraphArtifact.Status.ACTIVE}
            or artifact.completed_at is None
        ):
            raise FixtureSeedError(
                "fixture database topology has a foreign graph reference"
            )
        result[artifact.pk] = artifact
    return result


def _run_is_terminal(run) -> bool:
    from apps.knowledge_graph.models import GraphBuildRun

    valid = {
        (GraphBuildRun.Stage.FAILED, GraphBuildRun.Status.FAILED),
        (GraphBuildRun.Stage.SUPERSEDED, GraphBuildRun.Status.CANCELLED),
        (GraphBuildRun.Stage.STALE, GraphBuildRun.Status.CANCELLED),
    }
    return (
        (run.stage, run.status) in valid
        and run.finished_at is not None
        and run.lease_owner == ""
        and run.lease_expires_at is None
    )


def _identity_matches_artifact(run, artifact) -> bool:
    fields = (
        "rebuild_request_id",
        "evaluation_only",
        "orchestration_version",
        "build_key",
        "build_generation",
        "scope_type",
        "scope_id",
        "source_hash",
        "ontology_version",
        "extractor_version",
        "resolver_version",
        "filter_policy_version",
        "embedding_model_signature",
        "ontology_checksum",
        "filter_policy_checksum",
        "resolution_config_checksum",
        "assembly_version",
        "assembly_config_checksum",
    )
    return all(getattr(run, field) == getattr(artifact, field) for field in fields)


def _load_runs(context: GraphContext, artifacts):
    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun

    expected_request_ids = set(context.expected_requests.values())
    runs = tuple(
        GraphBuildRun.objects.filter(
            Q(artifact_id__in=artifacts)
            | Q(rebuild_request_id__in=expected_request_ids)
            | Q(
                scope_type=GraphArtifact.ScopeType.COLLECTION,
                scope_id__in=map(str, context.collection_ids),
            )
            | Q(
                scope_type=GraphArtifact.ScopeType.DOCUMENT,
                scope_id__in=map(str, context.document_collections),
            )
        ).order_by("pk")[:5_001]
    )
    if len(runs) > 5_000:
        raise FixtureSeedError(
            "fixture database topology has a foreign graph reference"
        )
    by_artifact = {}
    for run in runs:
        collection_id = _scope_collection(run, context)
        artifact = artifacts.get(run.artifact_id)
        if (
            collection_id is None
            or run.rebuild_request_id != context.expected_requests.get(collection_id)
            or run.evaluation_only is not True
            or not _run_is_terminal(run)
            or (artifact is not None and not _identity_matches_artifact(run, artifact))
            or (run.artifact_id is not None and artifact is None)
        ):
            raise FixtureSeedError(
                "fixture database topology has a foreign graph reference"
            )
        if artifact is not None:
            by_artifact.setdefault(artifact.pk, []).append(run)
    if any(len(by_artifact.get(artifact_id, ())) != 1 for artifact_id in artifacts):
        raise FixtureSeedError(
            "fixture database topology has an incoherent graph occurrence"
        )
    return runs, by_artifact


def _validate_succeeded_requests(requests, artifacts, by_artifact) -> None:
    from apps.knowledge_graph.models import (
        GraphArtifact,
        GraphBuildRun,
        GraphRebuildRequest,
    )
    from apps.knowledge_graph.models.artifacts import _activation_audit_values

    for request in requests.values():
        if request.status != GraphRebuildRequest.Status.SUCCEEDED:
            continue
        candidates = [
            artifact
            for artifact in artifacts.values()
            if artifact.rebuild_request_id == request.pk
            and artifact.scope_type == GraphArtifact.ScopeType.COLLECTION
            and artifact.status == GraphArtifact.Status.SUPERSEDED
        ]
        if len(candidates) != 1:
            raise FixtureSeedError(
                "fixture database topology has an incoherent graph occurrence"
            )
        artifact = candidates[0]
        run = by_artifact[artifact.pk][0]
        if (
            run.stage != GraphBuildRun.Stage.SUPERSEDED
            or run.status != GraphBuildRun.Status.CANCELLED
            or run.stage_marker.get("evaluation_completed") is not True
            or any(
                getattr(request, field) != value
                for field, value in _activation_audit_values(artifact, run).items()
            )
        ):
            raise FixtureSeedError(
                "fixture database topology has an incoherent graph occurrence"
            )


def _validate_dependents(context: GraphContext, artifacts) -> None:
    from apps.knowledge_graph.models import (
        CollectionArtifactInput,
        CollectionEntity,
        EntityMention,
    )

    artifact_ids = set(artifacts)
    if (
        EntityMention.objects.filter(chunk_id__in=context.chunk_ids)
        .exclude(artifact_id__in=artifact_ids)
        .exists()
    ):
        raise FixtureSeedError(
            "fixture database topology has a foreign graph reference"
        )
    for model in (CollectionArtifactInput, CollectionEntity):
        if (
            model.objects.filter(collection_id__in=context.collection_ids)
            .exclude(artifact_id__in=artifact_ids)
            .exists()
        ):
            raise FixtureSeedError(
                "fixture database topology has a foreign graph reference"
            )
    if (
        CollectionArtifactInput.objects.filter(collection_id__in=context.collection_ids)
        .exclude(artifact__collection_scope_id=F("collection_id"))
        .exists()
        or CollectionEntity.objects.filter(collection_id__in=context.collection_ids)
        .exclude(artifact__collection_scope_id=F("collection_id"))
        .exists()
    ):
        raise FixtureSeedError(
            "fixture database topology has a foreign graph reference"
        )


def validate_occurrences(
    context: GraphContext, requests, *, post_cleanup: bool
) -> None:
    artifacts = _load_artifacts(context)
    _runs, by_artifact = _load_runs(context, artifacts)
    if not post_cleanup:
        _validate_succeeded_requests(requests, artifacts, by_artifact)
    elif artifacts:
        raise FixtureSeedError(
            "fixture database topology has a foreign graph reference"
        )
    _validate_dependents(context, artifacts)
