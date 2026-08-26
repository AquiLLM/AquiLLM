from __future__ import annotations

from collections import defaultdict

from django.db.models import Q

from apps.knowledge_graph.extraction.windows import sanitize_graph_source_text
from apps.knowledge_graph.models import (
    CollectionEntity,
    CollectionEntityDocumentLink,
    CollectionRelation,
    CollectionRelationEvidence,
    DocumentEntityMention,
    GraphArtifact,
    GraphRebuildRequest,
)

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


def _status(active_artifact, request):
    if active_artifact is not None:
        return {
            "state": "ready",
            "error_code": None,
            "request_id": str(request.pk) if request is not None else None,
            "updated_at": _iso(active_artifact.updated_at),
        }
    if request is None:
        return {
            "state": "empty",
            "error_code": None,
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
    base = {
        "collection_id": str(collection.pk),
        "artifact_id": str(active_artifact.pk) if active_artifact is not None else None,
        "status": _status(active_artifact, request),
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
    node_rows = list(
        node_query.order_by("-retrieval_utility", "normalized_label", "pk").values(
            "pk",
            "label",
            "entity_type",
            "resolution_confidence",
            "retrieval_utility",
        )[: NODE_LIMIT + 1]
    )
    base["truncated"]["nodes"] = len(node_rows) > NODE_LIMIT
    node_rows = node_rows[:NODE_LIMIT]
    node_ids = tuple(row["pk"] for row in node_rows)
    node_evidence = _node_evidence(node_ids)

    edge_rows = list(
        CollectionRelation.objects.current()
        .filter(
            artifact=active_artifact,
            source_id__in=node_ids,
            target_id__in=node_ids,
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
    base["truncated"]["edges"] = len(edge_rows) > EDGE_LIMIT
    edge_rows = edge_rows[:EDGE_LIMIT]
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
