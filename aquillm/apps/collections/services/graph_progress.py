"""Content-free, collection-scoped progress for automatic document builds.

These counters describe build activity, not retrieval authority. Retrieval still
requires the complete collection artifact and its independently fenced projection.
"""

from collections import Counter

from django.db.models import CharField, Count, OuterRef, Subquery
from django.db.models.functions import Cast

from apps.documents.models import DESCENDED_FROM_DOCUMENT
from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun, OntologyVersion
from apps.knowledge_graph.services.failure_codes import DOCUMENT_CAPACITY_FAILURE_CODES

PUBLIC_DOCUMENT_FAILURE_CODES = DOCUMENT_CAPACITY_FAILURE_CODES | frozenset(
    {
        "document_build_failed",
        "corrupt_build_state",
    }
)


def selected_graph_ontology_identity(collection_id: int) -> dict | None:
    ontology_rows = OntologyVersion.objects.filter(kind="graph", status="active")
    selected = list(
        ontology_rows.filter(metadata__collection_id=collection_id).values(
            "version", "checksum"
        )[:2]
    )
    if not selected:
        selected = list(
            ontology_rows.exclude(metadata__has_key="collection_id").values(
                "version", "checksum"
            )[:2]
        )
    return selected[0] if len(selected) == 1 else None


def document_graph_progress(collection_id: int) -> dict:
    progress = dict.fromkeys(
        ("total", "ingesting", "pending", "building", "active", "failed"), 0
    )
    failures = Counter()
    selected = selected_graph_ontology_identity(collection_id)
    latest_error = (
        GraphBuildRun.objects.filter(
            artifact_id=OuterRef("pk"),
            evaluation_only=False,
        )
        .order_by("-attempt", "-pk")
        .values("error_code")[:1]
    )
    artifacts = GraphArtifact.objects.filter(
        scope_type="document",
        scope_id=Cast(OuterRef("id"), CharField()),
        source_hash=OuterRef("full_text_hash"),
        evaluation_only=False,
    ).order_by("-build_generation", "-pk")
    if selected is not None:
        artifacts = artifacts.filter(
            ontology_version=selected["version"],
            ontology_checksum=selected["checksum"],
        )
    artifacts = artifacts.annotate(latest_error=Subquery(latest_error))
    for model in DESCENDED_FROM_DOCUMENT:
        # Group in SQL: neither document text nor per-document histories leave DB.
        groups = (
            model.objects.filter(collection_id=collection_id)
            .order_by()
            .annotate(
                graph_state=Subquery(artifacts.values("status")[:1]),
                graph_error=Subquery(artifacts.values("latest_error")[:1]),
            )
            .values("ingestion_complete", "graph_state", "graph_error")
            .annotate(count=Count("pk"))
        )
        for group in groups.iterator(chunk_size=100):
            count = group["count"]
            progress["total"] += count
            state = group["graph_state"]
            if not group["ingestion_complete"]:
                progress["ingesting"] += count
            elif state in ("active", "building", "failed"):
                progress[state] += count
                if state == "failed":
                    code = group["graph_error"]
                    failures[
                        code
                        if code in PUBLIC_DOCUMENT_FAILURE_CODES
                        else "document_build_failed"
                    ] += count
            else:
                progress["pending"] += count
    progress["failures"] = [
        {"code": code, "count": count} for code, count in sorted(failures.items())
    ]
    return progress
