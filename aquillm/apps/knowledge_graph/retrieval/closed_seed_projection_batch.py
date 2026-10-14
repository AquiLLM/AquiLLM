"""Private bounded cross-projection seed lookup with fresh authority fences."""

from apps.collections.services.retrieval_authorization import revalidate_retrieval_authorization_context
from apps.knowledge_graph.projection.identifiers import ProjectionIdentifierDomain
from .graph_seed_reads_v2 import seed_query_rows
from .extended_seed_repository import _authority_query, _chunk_documents, _seed_queries


def load_seed_identities_batch(
    repository, *, requests, authorization, codec, max_rows, read_budget=None,
):
    """One unpublished bounded read, retaining fresh authority at both ends."""
    if (type(requests) is not tuple or not 1 <= len(requests) <= 64
            or not 1 <= max_rows <= 4999):
        raise ValueError("extended seed batch exceeds its hard cap")
    requested, owners = {}, {}
    for authority, chunks in requests:
        if authority.projection_id in owners or type(chunks) is not tuple or not chunks:
            raise ValueError("extended seed batch repeats a projection")
        owners[authority.projection_id] = authority
        documents = dict(authority.documents)
        for pk, document in chunks:
            if type(pk) is not int or pk <= 0 or pk in requested or document not in documents:
                raise ValueError("extended seed batch has invalid ownership")
            requested[pk] = document
    if not 1 <= len(requested) <= 128:
        raise ValueError("extended seed batch exceeds its chunk cap")

    def validate():
        current = revalidate_retrieval_authorization_context(context=authorization)
        if (not set(requested.values()) <= set(current.document_ids)
                or not {a.collection_id for a in owners.values()} <= set(current.collection_ids)):
            raise ValueError("extended seed authority is stale or unauthorized")
        queries = tuple(_authority_query(authority=a, using=repository.using).order_by()
                        .values_list("pk", flat=True) for a in owners.values())
        query = queries[0].union(*queries[1:], all=True) if len(queries) > 1 else queries[0]
        rows = seed_query_rows(query, maximum=len(owners), budget=read_budget)
        if len(rows) != len(owners) or set(rows) != set(owners):
            raise ValueError("extended seed authority is stale or unauthorized")

    validate()
    if _chunk_documents(chunk_ids=tuple(requested), using=repository.using,
                        read_budget=read_budget) != requested:
        raise ValueError("extended seed chunk ownership is stale")
    limit = max_rows
    if read_budget is not None:
        # Retain space for the final full-scope policy read, its duplicate
        # sentinel, exact projection-state rows and the seed overflow sentinel.
        final_authority_rows = (len(authorization.selected_collection_ids)
                                + len(authorization.selected_document_ids) + 1)
        limit = min(limit, read_budget.max_source_rows - read_budget.source_rows
                    - final_authority_rows - len(owners) - 1)
        if limit < 1:
            from .topology.frontier_contracts_v2 import FrontierBudgetExceeded
            raise FrontierBudgetExceeded("extended seed row allowance exhausted")
    queries = tuple(query for authority, chunks in requests
                    for query in _seed_queries(authority=authority, chunks=chunks,
                        documents=dict(authority.documents), using=repository.using))
    combined = queries[0].union(*queries[1:], all=True) if len(queries) > 1 else queries[0]
    rows = seed_query_rows(combined.order_by("seed_chunk_id", "collection_entity_id", "canonical_id"),
                           maximum=limit + 1, budget=read_budget)
    if len(rows) > limit:
        from .topology.frontier_contracts_v2 import FrontierBudgetExceeded
        raise FrontierBudgetExceeded("extended seed row sentinel reached")
    by_chunk = {pk: set() for pk in requested}
    chunk_authority = {pk: authority for authority, chunks in requests for pk, _ in chunks}
    for chunk_id, entity_id, canonical_id, automatic_link_count in rows:
        if ((automatic_link_count or 0) > 1 or type(chunk_id) is not int
                or chunk_id not in requested or type(entity_id) is not int or entity_id <= 0
                or (canonical_id is not None and (type(canonical_id) is not int or canonical_id <= 0))):
            raise ValueError("extended seed identity is invalid")
        by_chunk[chunk_id].add(codec.encode(
            ProjectionIdentifierDomain.ENTITY if canonical_id is None
            else ProjectionIdentifierDomain.AUTOMATIC_CANONICAL_IDENTITY,
            generation=chunk_authority[chunk_id].generation_id,
            source=entity_id if canonical_id is None else canonical_id,
        ).value)
    validate()
    return {authority.projection_id: {pk: tuple(sorted(by_chunk[pk])) for pk, _ in chunks}
            for authority, chunks in requests}
