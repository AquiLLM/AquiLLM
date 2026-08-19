"""Bounded, privacy-safe lifecycle inspection for operator commands."""

from __future__ import annotations

import re
import uuid
from math import isfinite
from time import monotonic, sleep

from django.core.exceptions import ValidationError
from django.db import close_old_connections
from django.db.models import Q

_MAX_INSPECTION_ROWS = 100
_TERMINAL_REQUEST_STATUSES = frozenset({"succeeded", "failed", "partial"})
_SAFE_CODE = re.compile(r"^[a-z][a-z0-9_]{0,127}$")


def _private_code(value: object) -> str:
    if value == "":
        return ""
    return value if type(value) is str and _SAFE_CODE.fullmatch(value) else "invalid"


def _validate_filters(
    document_id: uuid.UUID | None,
    collection_id: int | None,
    request_id: uuid.UUID | None,
    wait: bool,
    timeout_seconds: float,
) -> None:
    if document_id is not None and type(document_id) is not uuid.UUID:
        raise ValueError("document_id must be an exact UUID")
    if collection_id is not None and (
        type(collection_id) is not int or not 0 < collection_id < 2**63
    ):
        raise ValueError("collection_id must be a positive database integer")
    if request_id is not None and type(request_id) is not uuid.UUID:
        raise ValueError("request_id must be an exact UUID")
    if type(wait) is not bool:
        raise ValueError("wait must be a boolean")
    if wait and request_id is None:
        raise ValueError("wait requires a request_id")
    if (
        type(timeout_seconds) not in (int, float)
        or not isfinite(float(timeout_seconds))
        or not 0 < float(timeout_seconds) <= 3_600
    ):
        raise ValueError("timeout_seconds must be finite and in (0, 3600]")


def _raise_if_wait_deadline_elapsed(deadline: float) -> None:
    if monotonic() >= deadline:
        raise TimeoutError("timed out validating rebuild request")


def _validate_success_activation_page(
    requests: tuple[object, ...],
    *,
    deadline: float,
) -> None:
    """Validate one bounded request page with set-wise occurrence loading."""

    if not requests:
        return
    _raise_if_wait_deadline_elapsed(deadline)
    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun
    from apps.knowledge_graph.models.artifacts import _activation_audit_values
    from apps.knowledge_graph.services.builds import (
        _evaluation_occurrence_completed,
        _production_occurrence_completed,
    )

    for request in requests:
        if not request._activation_audit_is_complete():
            raise RuntimeError("operator-wide child lacks its exact activation")
    artifact_ids = tuple(request.activated_artifact_pk for request in requests)
    artifacts = GraphArtifact.objects.filter(pk__in=artifact_ids).in_bulk()
    request_ids = tuple(request.pk for request in requests)
    runs_by_artifact: dict[int, list[object]] = {}
    for run in GraphBuildRun.objects.filter(
        artifact_id__in=tuple(artifacts),
        rebuild_request_id__in=request_ids,
    ).order_by("artifact_id", "pk"):
        runs_by_artifact.setdefault(run.artifact_id, []).append(run)
    for request in requests:
        _raise_if_wait_deadline_elapsed(deadline)
        artifact = artifacts.get(request.activated_artifact_pk)
        if artifact is None:
            # Successful request audit is immutable and remains authoritative
            # after retention removes its exact graph occurrence.
            continue
        expected_scope = (
            GraphArtifact.ScopeType.DOCUMENT
            if request.scope_type == request.ScopeType.DOCUMENT
            else GraphArtifact.ScopeType.COLLECTION
        )
        if (
            artifact.rebuild_request_id != request.pk
            or artifact.scope_type != expected_scope
            or artifact.scope_id != request.scope_id
            or artifact.evaluation_only is not request.evaluation_only
        ):
            raise RuntimeError("operator-wide child activation is outside its request")
        if request.scope_type == request.ScopeType.DOCUMENT:
            source_matches = len(
                request.requested_documents
            ) == 1 and artifact.source_hash == request.requested_documents[0].get(
                "source_hash"
            )
            build_kind = GraphBuildRun.BuildKind.DOCUMENT
        else:
            source_matches = (
                bool(request.expected_aggregate_signature)
                and artifact.source_hash == request.expected_aggregate_signature
            )
            build_kind = GraphBuildRun.BuildKind.COLLECTION
        runs = runs_by_artifact.get(artifact.pk, [])
        if not source_matches or len(runs) != 1:
            raise RuntimeError("operator-wide child lacks its exact activation")
        run = runs[0]
        if run.pk != request.activated_run_pk or run.build_kind != build_kind:
            raise RuntimeError("operator-wide child activation run changed")
        completed = (
            _evaluation_occurrence_completed(
                artifact,
                run,
                build_kind=build_kind,
            )
            if request.evaluation_only
            else _production_occurrence_completed(
                artifact,
                run,
                build_kind=build_kind,
                allow_historical=True,
            )
        )
        if not completed:
            raise RuntimeError("operator-wide child activation is not terminal")
        expected_audit = _activation_audit_values(artifact, run)
        if any(
            getattr(request, field) != value for field, value in expected_audit.items()
        ):
            raise RuntimeError("operator-wide child activation audit changed")


def _wait_for_request(request_id: uuid.UUID, timeout_seconds: float):
    from apps.knowledge_graph.models import GraphRebuildRequest
    from apps.knowledge_graph.services.builds import _effective_rebuild_request

    deadline = monotonic() + float(timeout_seconds)
    while True:
        close_old_connections()
        request = GraphRebuildRequest.objects.filter(pk=request_id).first()
        if request is None:
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise TimeoutError("rebuild request was not visible before timeout")
            sleep(min(0.25, remaining))
            continue
        effective = _effective_rebuild_request(request)
        if effective.status in _TERMINAL_REQUEST_STATUSES:
            if effective.error_code == "resnapshot_pending":
                remaining = deadline - monotonic()
                if remaining <= 0:
                    raise TimeoutError("timed out waiting for rebuild reconciliation")
                sleep(min(0.25, remaining))
                continue
            if effective.status != GraphRebuildRequest.Status.SUCCEEDED:
                raise RuntimeError(f"rebuild request ended {effective.status}")
            if effective.scope_type == GraphRebuildRequest.ScopeType.ALL:
                if (
                    not effective.enumeration_complete
                    or effective.completed_collection_count
                    != effective.expected_child_count
                    or effective.failed_collection_count != 0
                ):
                    raise RuntimeError(
                        "operator-wide rebuild lacks exact successful children"
                    )
                leaves = GraphRebuildRequest.objects.filter(
                    Q(parent_request_id=effective.pk)
                    | Q(lineage_root__parent_request_id=effective.pk),
                    successor_request__isnull=True,
                ).order_by("pk")
                child_count = 0
                after_pk = None
                while True:
                    _raise_if_wait_deadline_elapsed(deadline)
                    page_query = leaves
                    if after_pk is not None:
                        page_query = page_query.filter(pk__gt=after_pk)
                    page = tuple(page_query[:_MAX_INSPECTION_ROWS])
                    if not page:
                        break
                    if any(
                        child.status != GraphRebuildRequest.Status.SUCCEEDED
                        for child in page
                    ):
                        raise RuntimeError(
                            "operator-wide rebuild lacks exact successful children"
                        )
                    _validate_success_activation_page(page, deadline=deadline)
                    child_count += len(page)
                    after_pk = page[-1].pk
                if child_count != effective.expected_child_count:
                    raise RuntimeError("operator-wide rebuild child count changed")
            else:
                try:
                    effective._validate_success_activation()
                except ValidationError as exc:
                    raise RuntimeError(
                        "rebuild request terminal state lacks its exact activation"
                    ) from exc
            return effective
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise TimeoutError("timed out waiting for rebuild request")
        sleep(min(0.25, remaining))


def _scope_filter(document_id: uuid.UUID | None, collection_id: int | None) -> Q:
    scope = Q()
    if document_id is not None:
        scope |= Q(scope_type="document", scope_id=str(document_id))
    if collection_id is not None:
        scope |= Q(scope_type="collection", scope_id=str(collection_id))
    return scope


def inspect_graph_state(
    *,
    document_id: uuid.UUID | None = None,
    collection_id: int | None = None,
    request_id: uuid.UUID | None = None,
    wait: bool = False,
    timeout_seconds: float = 60.0,
) -> dict[str, object]:
    """Return bounded IDs, versions, statuses, and aggregate counts only."""

    _validate_filters(document_id, collection_id, request_id, wait, timeout_seconds)
    from apps.knowledge_graph.models import (
        CollectionEntity,
        CollectionRelation,
        DocumentEntity,
        EntityMention,
        GraphArtifact,
        GraphBuildRun,
        GraphRebuildRequest,
    )

    request = None
    if wait:
        request = _wait_for_request(request_id, timeout_seconds)
    elif request_id is not None:
        request = GraphRebuildRequest.objects.filter(pk=request_id).first()
        if request is None:
            raise LookupError("rebuild request does not exist")
    effective_request = request
    if request is not None:
        from apps.knowledge_graph.services.builds import _effective_rebuild_request

        effective_request = _effective_rebuild_request(request)

    artifact_query = GraphArtifact.objects.order_by("-created_at", "-pk")
    build_query = GraphBuildRun.objects.order_by("-created_at", "-pk")
    scope = _scope_filter(document_id, collection_id)
    if document_id is not None or collection_id is not None:
        artifact_query = artifact_query.filter(scope)
        build_query = build_query.filter(scope)
    if request_id is not None:
        assert request is not None
        lineage_root_id = request.lineage_root_id or request.pk
        if request.scope_type == GraphRebuildRequest.ScopeType.ALL:
            correlation = Q(rebuild_request__parent_request_id=lineage_root_id) | Q(
                rebuild_request__lineage_root__parent_request_id=(lineage_root_id)
            )
        else:
            correlation = Q(rebuild_request_id=lineage_root_id) | Q(
                rebuild_request__lineage_root_id=lineage_root_id
            )
        artifact_query = artifact_query.filter(correlation)
        build_query = build_query.filter(correlation)

    artifact_count = artifact_query.count()
    build_count = build_query.count()
    stale_count = artifact_query.filter(
        status__in=(GraphArtifact.Status.STALE, GraphArtifact.Status.SUPERSEDED)
    ).count()
    active_artifacts = artifact_query.filter(status=GraphArtifact.Status.ACTIVE)
    evidence_count = (
        EntityMention.objects.filter(artifact__in=active_artifacts).count()
        + DocumentEntity.objects.filter(artifact__in=active_artifacts).count()
        + CollectionEntity.objects.filter(artifact__in=active_artifacts).count()
        + CollectionRelation.objects.filter(artifact__in=active_artifacts).count()
    )
    if effective_request is not None:
        failure_count = (
            effective_request.terminal_failure_count
            + effective_request.failed_collection_count
        )
    else:
        failure_count = (
            build_query.filter(
                Q(status=GraphBuildRun.Status.FAILED)
                | Q(
                    status=GraphBuildRun.Status.CANCELLED,
                    stage=GraphBuildRun.Stage.STALE,
                )
                | (Q(status=GraphBuildRun.Status.CANCELLED) & ~Q(error_code=""))
            )
            .exclude(stage_marker__evaluation_completed=True)
            .count()
        )

    artifact_rows = tuple(
        artifact_query.values(
            "pk",
            "scope_type",
            "scope_id",
            "status",
            "ontology_version",
            "extractor_version",
            "resolver_version",
            "filter_policy_version",
            "assembly_version",
            "rebuild_request_id",
            "evaluation_only",
        )[: _MAX_INSPECTION_ROWS + 1]
    )
    build_rows = tuple(
        build_query.values(
            "pk",
            "artifact_id",
            "scope_type",
            "scope_id",
            "stage",
            "status",
            "error_code",
            "rebuild_request_id",
            "evaluation_only",
        )[: _MAX_INSPECTION_ROWS + 1]
    )
    truncated = (
        artifact_count > _MAX_INSPECTION_ROWS or build_count > _MAX_INSPECTION_ROWS
    )
    artifact_rows = artifact_rows[:_MAX_INSPECTION_ROWS]
    build_rows = build_rows[:_MAX_INSPECTION_ROWS]
    artifact_ids = tuple(row["pk"] for row in artifact_rows)
    return {
        "request_id": str(request_id) if request_id is not None else None,
        "effective_request_id": (
            str(effective_request.pk) if effective_request is not None else None
        ),
        "status": effective_request.status if effective_request is not None else None,
        "artifact_count": artifact_count,
        "build_count": build_count,
        "stale_count": stale_count,
        "active_evidence_count": evidence_count,
        "failure_count": failure_count,
        "document_count": (
            effective_request.document_count if effective_request is not None else None
        ),
        "completed_document_count": (
            effective_request.completed_document_count
            if effective_request is not None
            else None
        ),
        "collection_count": (
            effective_request.collection_count
            if effective_request is not None
            else None
        ),
        "completed_collection_count": (
            effective_request.completed_collection_count
            if effective_request is not None
            else None
        ),
        "failed_collection_count": (
            effective_request.failed_collection_count
            if effective_request is not None
            else None
        ),
        "truncated": truncated,
        "artifact_ids": artifact_ids,
        "build_ids": tuple(row["pk"] for row in build_rows),
        "artifacts": tuple(
            {
                key: (str(value) if key == "rebuild_request_id" and value else value)
                for key, value in row.items()
            }
            for row in artifact_rows
        ),
        "builds": tuple(
            {
                key: (
                    str(value)
                    if key == "rebuild_request_id" and value
                    else _private_code(value)
                    if key == "error_code"
                    else value
                )
                for key, value in row.items()
            }
            for row in build_rows
        ),
    }


__all__ = ["inspect_graph_state"]
