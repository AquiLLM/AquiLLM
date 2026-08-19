"""Request-side cleanup fences for synthetic fixture graph audit rows."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from django.db.models import Q

from .fixture_manifest import ResolvedFixtureManifest
from .fixture_seed_contract import PHYSICAL_BINDINGS, FixtureSeedError
from .fixture_seed_query import bounded_rows, require_exact_unique_rows


@dataclass(frozen=True, slots=True)
class GraphContext:
    expected_requests: dict[int, UUID]
    collection_ids: frozenset[int]
    document_collections: dict[UUID, int]
    document_hashes: dict[UUID, str]
    document_pkids: dict[str, int]
    chunk_ids: frozenset[int]


def physical_labels(resolved: ResolvedFixtureManifest) -> dict[int, str]:
    result = {}
    for symbol, binding in resolved.collections.items():
        label = PHYSICAL_BINDINGS.get(symbol)
        if label is None or result.setdefault(binding.collection_id, label) != label:
            raise FixtureSeedError("fixture database topology is not exact")
    if set(result.values()) != set(PHYSICAL_BINDINGS.values()):
        raise FixtureSeedError("fixture database topology is not exact")
    return result


def expected_request_by_collection(
    resolved: ResolvedFixtureManifest,
) -> dict[int, UUID]:
    return {
        binding.collection_id: binding.rebuild_request_id
        for binding in resolved.collections.values()
        if binding.authorized and binding.rebuild_request_id is not None
    }


def graph_context(
    resolved: ResolvedFixtureManifest, *, rows_present: bool
) -> GraphContext:
    from apps.documents.models import RawTextDocument

    collections = frozenset(physical_labels(resolved))
    document_collections = {
        binding.document_id: binding.collection_id
        for binding in resolved.documents.values()
    }
    document_hashes = {
        binding.document_id: binding.full_text_sha256
        for binding in resolved.documents.values()
    }
    document_ids = frozenset(document_collections)
    persisted = bounded_rows(
        RawTextDocument.objects.filter(id__in=document_ids).values_list("id", "pkid"),
        len(document_ids),
        order_by=("id", "pkid"),
    )
    if rows_present:
        mapping = require_exact_unique_rows(
            persisted,
            document_ids,
            key=lambda row: row[0],
        )
        document_pkids = {
            str(document_id): row[1] for document_id, row in mapping.items()
        }
    elif persisted:
        raise FixtureSeedError("fixture database topology is not exact")
    else:
        document_pkids = {}
    return GraphContext(
        expected_request_by_collection(resolved),
        collections,
        document_collections,
        document_hashes,
        document_pkids,
        frozenset(binding.chunk_id for binding in resolved.chunks.values()),
    )


def _request_is_terminal(request) -> bool:
    from apps.knowledge_graph.models import GraphRebuildRequest

    if request.status in {
        GraphRebuildRequest.Status.QUEUED,
        GraphRebuildRequest.Status.RUNNING,
    }:
        return False
    if request.completed_at is None or request.document_publication_state == "pending":
        return False
    if request.collection_publication_state == "pending":
        return False
    return not (
        request.status == GraphRebuildRequest.Status.PARTIAL
        and request.error_code in {"resnapshot_pending", "resnapshot_churn"}
    )


def validate_requests(context: GraphContext) -> dict[UUID, object]:
    from apps.documents.models import RawTextDocument
    from apps.knowledge_graph.models import GraphRebuildRequest

    expected_ids = set(context.expected_requests.values())
    requests = tuple(
        GraphRebuildRequest.objects.filter(
            Q(pk__in=expected_ids)
            | Q(scope_type="collection", scope_id__in=map(str, context.collection_ids))
        ).order_by("pk")[:5_001]
    )
    if len(requests) > 5_000:
        raise FixtureSeedError(
            "fixture database topology has a foreign graph reference"
        )
    result = {}
    reverse = {
        request_id: collection_id
        for collection_id, request_id in context.expected_requests.items()
    }
    for request in requests:
        collection_id = reverse.get(request.pk)
        expected_documents = {
            str(document_id): context.document_hashes[document_id]
            for document_id, physical_id in context.document_collections.items()
            if physical_id == collection_id
        }
        snapshots = request.requested_documents
        if (
            collection_id is None
            or request.scope_type != GraphRebuildRequest.ScopeType.COLLECTION
            or request.scope_id != str(collection_id)
            or request.evaluation_only is not True
            or request.collection_count != 1
            or request.document_count != len(expected_documents)
            or type(snapshots) is not list
            or len(snapshots) != len(expected_documents)
            or not _request_is_terminal(request)
        ):
            raise FixtureSeedError(
                "fixture database topology has a foreign graph reference"
            )
        seen = set()
        for snapshot in snapshots:
            document_id = (
                snapshot.get("document_id") if type(snapshot) is dict else None
            )
            if (
                type(snapshot) is not dict
                or set(snapshot)
                != {
                    "document_id",
                    "document_pkid",
                    "model_label",
                    "collection_id",
                    "source_hash",
                }
                or type(document_id) is not str
                or document_id in seen
                or expected_documents.get(document_id) != snapshot["source_hash"]
                or snapshot["collection_id"] != collection_id
                or snapshot["model_label"] != RawTextDocument._meta.label_lower
                or type(snapshot["document_pkid"]) is not int
                or (
                    context.document_pkids
                    and context.document_pkids.get(document_id)
                    != snapshot["document_pkid"]
                )
            ):
                raise FixtureSeedError(
                    "fixture database topology has a foreign graph reference"
                )
            seen.add(document_id)
        if seen != set(expected_documents):
            raise FixtureSeedError(
                "fixture database topology has a foreign graph reference"
            )
        result[request.pk] = request
    return result
