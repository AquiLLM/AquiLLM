"""Canonical occurrence locking for rebuild failure reconciliation."""

from __future__ import annotations


def locked_completed_document_count(request: object) -> int:
    """Fence request-owned document occurrences before lifecycle validation."""

    from django.db import connection

    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun
    from apps.knowledge_graph.services.builds import (
        CorruptBuildError,
        _completed_request_document_artifacts,
    )

    if not connection.in_atomic_block:
        raise RuntimeError("rebuild failure occurrence locking requires a transaction")
    maximum = request.document_count
    artifacts = tuple(
        GraphArtifact.objects.select_for_update(no_key=True)
        .filter(
            rebuild_request_id=request.pk,
            scope_type=GraphArtifact.ScopeType.DOCUMENT,
        )
        .order_by("pk")[: maximum + 1]
    )
    if len(artifacts) > maximum:
        raise CorruptBuildError("request owns surplus document artifacts")
    runs = tuple(
        GraphBuildRun.objects.select_for_update()
        .filter(
            rebuild_request_id=request.pk,
            build_kind=GraphBuildRun.BuildKind.DOCUMENT,
        )
        .order_by("pk")[: maximum + 1]
    )
    if len(runs) > maximum:
        raise CorruptBuildError("request owns surplus document build runs")
    return len(
        _completed_request_document_artifacts(
            request,
            tuple(request.requested_documents),
            for_update=False,
        )
    )


__all__ = ["locked_completed_document_count"]
