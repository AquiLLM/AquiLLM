"""Set-wise current lineage progress, separate from immutable request audits."""

from django.db.models import Count, Q, Sum

_STATUSES = ("queued", "running", "succeeded", "partial", "failed")
_AUDIT_COUNTERS = (
    "document_count",
    "completed_document_count",
    "collection_count",
    "completed_collection_count",
    "failed_collection_count",
)


def request_audit_counters(request):
    """Preserve the legacy top-level counters of the inspected request."""
    return {field: getattr(request, field, None) for field in _AUDIT_COUNTERS}


def _live_status(request, totals, *, enumeration_complete):
    if totals["running"]:
        return "running"
    if totals["queued"]:
        return "queued"
    if totals["resnapshot_pending_count"]:
        return "pending"
    if not enumeration_complete:
        return (
            request.status
            if request.status in {"queued", "running", "failed"}
            else "pending"
        )
    if (
        request.scope_type == "all"
        and totals["effective_request_count"] < request.expected_child_count
    ):
        return "pending"
    if totals["partial"] or (totals["failed"] and totals["succeeded"]):
        return "partial"
    if totals["failed"]:
        return "failed"
    if totals["succeeded"]:
        return "succeeded"
    return request.status


def request_progress(request, artifact_query):
    """Aggregate current leaves without loading snapshots or replacement history.

    Artifact counts use the inspection's existing scope/lineage filters. Historical
    activations count retained activated occurrences (including current ones), not
    a lifetime total after retention has deleted artifacts.
    Pending resnapshots include every reconcilable leaf; churn is a named subset.
    """
    if request is None:
        return None
    from apps.knowledge_graph.models import GraphRebuildRequest
    from apps.knowledge_graph.services.builds import (
        _RESNAPSHOT_CHURN_ERROR,
        _RESNAPSHOT_RECONCILABLE_ERRORS,
    )

    if request.scope_type == GraphRebuildRequest.ScopeType.ALL:
        leaves = GraphRebuildRequest.objects.filter(
            Q(parent_request_id=request.pk)
            | Q(lineage_root__parent_request_id=request.pk),
            successor_request__isnull=True,
        )
        enumeration_complete = request.enumeration_complete
    else:
        # inspect_graph_state already follows the scoped request's successor chain.
        leaves = GraphRebuildRequest.objects.filter(pk=request.pk)
        enumeration_complete = True
    totals = leaves.aggregate(
        document_count=Sum("document_count", default=0),
        completed_document_count=Sum("completed_document_count", default=0),
        failed_document_count=Sum("terminal_failure_count", default=0),
        collection_count=Sum("collection_count", default=0),
        effective_request_count=Count("pk"),
        resnapshot_pending_count=Count(
            "pk", filter=Q(error_code__in=_RESNAPSHOT_RECONCILABLE_ERRORS)
        ),
        resnapshot_churn_count=Count(
            "pk", filter=Q(error_code=_RESNAPSHOT_CHURN_ERROR)
        ),
        **{status: Count("pk", filter=Q(status=status)) for status in _STATUSES},
        **{
            f"collection_{status}": Count(
                "pk", filter=Q(scope_type="collection", status=status)
            )
            for status in _STATUSES
        },
    )
    artifacts = artifact_query.aggregate(
        active_document_artifact_count=Count(
            "pk", filter=Q(status="active", scope_type="document")
        ),
        active_collection_artifact_count=Count(
            "pk", filter=Q(status="active", scope_type="collection")
        ),
        historical_activation_count=Count("pk", filter=Q(activated_at__isnull=False)),
    )
    return {
        **{
            field: totals[field]
            for field in (
                "document_count",
                "completed_document_count",
                "failed_document_count",
                "collection_count",
                "effective_request_count",
                "resnapshot_pending_count",
                "resnapshot_churn_count",
            )
        },
        "collection_status_counts": {
            status: totals[f"collection_{status}"] for status in _STATUSES
        },
        "enumeration_complete": enumeration_complete,
        "live_status": _live_status(
            request, totals, enumeration_complete=enumeration_complete
        ),
        **artifacts,
    }
