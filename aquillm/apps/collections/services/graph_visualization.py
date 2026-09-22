from __future__ import annotations

from collections import defaultdict

from django.db.models import Q
from django.utils import timezone

from apps.collections.services.graph_progress import (
    document_graph_progress,
    selected_graph_ontology_identity,
)
from apps.documents.models import DESCENDED_FROM_DOCUMENT
from apps.knowledge_graph.extraction.windows import sanitize_graph_source_text
from apps.knowledge_graph.models import (
    CollectionEntity,
    CollectionEntityDocumentLink,
    CollectionRelation,
    CollectionRelationEvidence,
    DocumentEntityMention,
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

NODE_LIMIT = 150
EDGE_LIMIT = 300
EVIDENCE_PER_EDGE_LIMIT = 3
EVIDENCE_PER_NODE_LIMIT = 3
EXCERPT_CHARACTER_LIMIT = 360


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


def _bounded_excerpt(value: str) -> str:
    safe = sanitize_graph_source_text(value).strip()
    if len(safe) <= EXCERPT_CHARACTER_LIMIT:
        return safe
    return safe[: EXCERPT_CHARACTER_LIMIT - 1].rstrip() + "…"


def _edge_evidence(edge_ids: tuple[int, ...]):
    grouped = defaultdict(list)
    if not edge_ids:
        return grouped
    rows = (
        CollectionRelationEvidence.objects.current()
        .filter(relation_id__in=edge_ids)
        .select_related(
            "relation_mention__chunk",
            "relation_mention__head",
            "relation_mention__tail",
        )
        .order_by("relation_id", "pk")
    )
    for row in rows:
        evidence = grouped[row.relation_id]
        if len(evidence) >= EVIDENCE_PER_EDGE_LIMIT:
            continue
        mention = row.relation_mention
        evidence.append(
            {
                "document_id": str(mention.document_id),
                "chunk_id": mention.chunk_id,
                "start": min(mention.head.start, mention.tail.start),
                "end": max(mention.head.end, mention.tail.end),
                "excerpt": _bounded_excerpt(mention.chunk.content),
            }
        )
    return grouped


def _node_evidence(node_ids: tuple[int, ...]):
    grouped = defaultdict(list)
    if not node_ids:
        return grouped
    links = list(
        CollectionEntityDocumentLink.objects.current()
        .filter(collection_entity_id__in=node_ids)
        .order_by("collection_entity_id", "pk")
        .values("collection_entity_id", "document_entity_id")[: NODE_LIMIT * 20]
    )
    document_nodes = defaultdict(list)
    for link in links:
        document_nodes[link["document_entity_id"]].append(link["collection_entity_id"])
    if not document_nodes:
        return grouped
    rows = (
        DocumentEntityMention.objects.filter(
            document_entity_id__in=document_nodes,
            status=DocumentEntityMention.Status.ACTIVE,
        )
        .select_related("mention__chunk")
        .order_by("document_entity_id", "mention_id")
    )
    scanned = 0
    for row in rows.iterator(chunk_size=500):
        scanned += 1
        if scanned > NODE_LIMIT * EVIDENCE_PER_NODE_LIMIT * 10:
            break
        mention = row.mention
        descriptor = {
            "document_id": str(mention.document_id),
            "chunk_id": mention.chunk_id,
            "start": mention.start,
            "end": mention.end,
            "excerpt": _bounded_excerpt(mention.chunk.content),
        }
        for node_id in document_nodes[row.document_entity_id]:
            evidence = grouped[node_id]
            if len(evidence) < EVIDENCE_PER_NODE_LIMIT:
                evidence.append(descriptor)
    return grouped


def collection_graph_envelope(collection, user, *, query: str = "") -> dict:
    active_artifact = _active_artifact(collection)
    request = _latest_request(collection)
    progress = document_graph_progress(collection.pk)
    attempt = (
        _current_collection_attempt(collection, progress)
        if active_artifact is None
        else None
    )
    base = {
        "collection_id": str(collection.pk),
        "artifact_id": str(active_artifact.pk) if active_artifact is not None else None,
        "status": _status(active_artifact, request, progress, attempt),
        "progress": progress,
        "permissions": {"can_rebuild": collection.user_can_edit(user)},
        "nodes": [],
        "edges": [],
        "truncated": {"nodes": False, "edges": False},
    }
    if active_artifact is None:
        return base

    node_query = CollectionEntity.objects.current().filter(
        collection=collection,
        artifact=active_artifact,
    )
    if query:
        node_query = node_query.filter(
            Q(label__icontains=query) | Q(entity_type__icontains=query)
        )
    node_fields = (
        "pk",
        "label",
        "entity_type",
        "resolution_confidence",
        "retrieval_utility",
    )
    eligible_node_ids = node_query.values("pk")
    edge_candidates = list(
        CollectionRelation.objects.current()
        .filter(
            artifact=active_artifact,
            source_id__in=eligible_node_ids,
            target_id__in=eligible_node_ids,
        )
        .order_by("-support_count", "-confidence", "pk")
        .values(
            "pk",
            "source_id",
            "target_id",
            "relation_type",
            "confidence",
            "support_count",
        )[: EDGE_LIMIT + 1]
    )
    base["truncated"]["edges"] = len(edge_candidates) > EDGE_LIMIT
    edge_rows = []
    connected_node_ids = []
    connected_node_id_set = set()
    for edge in edge_candidates[:EDGE_LIMIT]:
        additions = tuple(
            node_id
            for node_id in (edge["source_id"], edge["target_id"])
            if node_id not in connected_node_id_set
        )
        if len(connected_node_id_set) + len(additions) > NODE_LIMIT:
            base["truncated"]["edges"] = True
            continue
        edge_rows.append(edge)
        connected_node_ids.extend(additions)
        connected_node_id_set.update(additions)

    connected_by_id = {
        row["pk"]: row
        for row in node_query.filter(pk__in=connected_node_ids).values(*node_fields)
    }
    connected_rows = [
        connected_by_id[node_id]
        for node_id in connected_node_ids
        if node_id in connected_by_id
    ]
    remaining = NODE_LIMIT - len(connected_rows)
    other_rows = list(
        node_query.exclude(pk__in=connected_node_id_set)
        .order_by("-retrieval_utility", "normalized_label", "pk")
        .values(*node_fields)[: remaining + 1]
    )
    base["truncated"]["nodes"] = len(other_rows) > remaining
    node_rows = connected_rows + other_rows[:remaining]
    node_ids = tuple(row["pk"] for row in node_rows)
    node_evidence = _node_evidence(node_ids)

    evidence = _edge_evidence(tuple(row["pk"] for row in edge_rows))

    base["nodes"] = [
        {
            "id": f"entity:{row['pk']}",
            "label": row["label"],
            "entity_type": row["entity_type"],
            "confidence": row["resolution_confidence"],
            "retrieval_utility": row["retrieval_utility"],
            "evidence": node_evidence[row["pk"]],
        }
        for row in node_rows
    ]
    base["edges"] = [
        {
            "id": f"relation:{row['pk']}",
            "source": f"entity:{row['source_id']}",
            "target": f"entity:{row['target_id']}",
            "relation_type": row["relation_type"],
            "confidence": row["confidence"],
            "support_count": row["support_count"],
            "evidence": evidence[row["pk"]],
        }
        for row in edge_rows
    ]
    return base


__all__ = ["collection_graph_envelope"]
