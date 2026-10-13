"""Fresh batched metadata under the original source/operation deadline only."""

from django.db.models import Count


def capture_authority_check():
    from apps.documents.services.source_deadline import _DEADLINE, check_source_deadline
    from apps.documents.services.source_loading import SourcePreparationLimited
    from lib.retrieval.operation import Operation, _OPERATION

    bound, operation = _DEADLINE.get(), _OPERATION.get()
    chain, seen, cursor = [], set(), operation
    while cursor is not None:
        if type(cursor) is not Operation or id(cursor) in seen or len(chain) >= 64:
            raise TimeoutError("graph authority operation ancestry unavailable")
        seen.add(id(cursor))
        chain.append((cursor, cursor.budget, cursor.deadline, cursor.clock, cursor.parent))
        cursor = cursor.parent

    def identity():
        if _DEADLINE.get() is not bound or _OPERATION.get() is not operation:
            raise TimeoutError("graph authority original context changed")
        for original, budget, deadline, clock, parent in chain:
            if (original.budget is not budget or original.deadline != deadline
                    or original.clock is not clock or original.parent is not parent):
                raise TimeoutError("graph authority original operation changed")
        if bound is not None and operation is not None and bound[2] is not None:
            if operation.budget is not bound[2]:
                raise TimeoutError("graph authority original budget changed")

    def check():
        identity()
        if operation is not None and not operation.active():
            raise TimeoutError("graph authority original operation expired")
        try:
            check_source_deadline()
        except SourcePreparationLimited as error:
            raise TimeoutError("graph authority deadline") from error
        if operation is not None and not operation.active():
            raise TimeoutError("graph authority original operation expired")
        identity()  # Final reference checks invoke no clock/budget callbacks.

    check()
    return check


def checked_rows(query, check):
    check()
    for row in query.iterator(chunk_size=128):
        check()
        yield row
    check()


def fresh_metadata(rows, selected, *, using, check):
    from apps.knowledge_graph.models import (
        CollectionArtifactInput, CollectionGraphMembershipState, GraphArtifact,
    )

    collections = tuple(sorted({row.collection_id for row in rows}))
    artifact_ids = tuple(sorted({row.artifact_id for row in rows if row.artifact_id is not None}))
    states = {row.collection_id: row for row in checked_rows(
        CollectionGraphMembershipState.objects.using(using).filter(collection_id__in=collections), check,
    )}
    artifacts = {row.pk: row for row in checked_rows(
        GraphArtifact.objects.using(using).filter(pk__in=artifact_ids), check,
    )}
    counts = dict(checked_rows(
        GraphArtifact.objects.using(using).filter(
            scope_type="collection", collection_scope_id__in=collections,
            status="active", evaluation_only=False,
        ).values("collection_scope_id").annotate(_count=Count("pk"))
         .values_list("collection_scope_id", "_count"), check,
    ))
    inputs = {key: [] for key in artifact_ids}
    limit = len(artifact_ids) * 10_000 + 1
    query = CollectionArtifactInput.objects.using(using).filter(
        artifact_id__in=artifact_ids, document_id__in=selected,
    ).select_related("document_artifact").order_by("artifact_id", "document_id", "pk")[:limit]
    for row in checked_rows(query, check):
        bucket = inputs[row.artifact_id]
        bucket.append(row)
        if len(bucket) > 10_000:
            raise ValueError("graph authority input limit")
    check()
    return states, artifacts, counts, inputs
