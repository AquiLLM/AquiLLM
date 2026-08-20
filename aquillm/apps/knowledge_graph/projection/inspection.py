from __future__ import annotations

from apps.knowledge_graph.models import CollectionGraphProjection


def _size(value: object) -> int:
    if type(value) is not int or not 1 <= value <= 5_000:
        raise ValueError("page_size must be an integer in 1..5000")
    return value


def _collection(value: object) -> int:
    if type(value) is not int or value < 1:
        raise ValueError("collection_id must be a positive integer")
    return value


def inspect_projection_authority(
    *, collection_id: int | None, all_collections: bool, page_size: int
) -> dict[str, int]:
    from . import reconciler

    size = _size(page_size)
    if type(all_collections) is not bool:
        raise TypeError("all_collections must be exact")
    selected_collection = None if all_collections else _collection(collection_id)
    query = CollectionGraphProjection.objects.all()
    if selected_collection is not None:
        query = query.filter(collection_pk_snapshot=selected_collection)
    counts = {"ready_count": 0, "pending_count": 0, "failed_count": 0}
    for state in query.order_by("id").values_list("state", flat=True)[:size]:
        key = f"{state}_count"
        if key in counts:
            counts[key] += 1
    audit = reconciler.reconcile_graph_projections(
        page_size=size,
        dry_run=True,
        collection_id=selected_collection,
    )
    counts["drift_count"] = audit.drift_count
    counts["orphan_count"] = audit.orphan_count
    counts["replayed_count"] = audit.replayed_count
    return counts


__all__ = ["inspect_projection_authority"]
