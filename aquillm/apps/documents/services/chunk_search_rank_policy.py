"""Independent optional ranking treatments and private runtime fusion adapters."""

from django.conf import settings

from lib.embeddings.document_contracts import digest

from .candidate_rank_fusion import fuse_ranked_channels_v2
from .lexical_candidates import strong_exact_ids
from .lexical_projection import POLICY


def controls():
    rrf = getattr(settings, "RAG_RRF_ENABLED", False) is True
    return {
        "full_text": getattr(settings, "RAG_FULL_TEXT_ENABLED", False) is True,
        "rrf": rrf,
        "protected_exact": rrf
        and getattr(settings, "RAG_PROTECTED_EXACT_ENABLED", False) is True,
    }


def ranking_policy(scope):
    return {
        "version": "lexical-ranking-v2",
        "controls": controls(),
        "projection": POLICY,
        "fts_bindings": sorted(
            (pk, digest(binding)) for pk, binding in scope.fts_rows.items()
        ),
    }


def fusion_scope_key(
    scope, *, graph_authority="graph-disabled-v1", graph_policy="graph-disabled-v1"
):
    from .representation_store import representation_snapshot_digest
    from .retrieval_scope_identity import retrieval_scope_key

    return retrieval_scope_key(
        authorization_digest=getattr(
            scope.authorization,
            "authorization_context_signature",
            "unbound-lexical-only-v1",
        ),
        representation_digest=representation_snapshot_digest(scope.representation),
        dense_generation_digest=scope.dense.manifest_sha256
        if scope.dense
        else "dense-unavailable-v1",
        embedding_space_key=scope.query_compatibility,
        graph_authority_digest=graph_authority,
        graph_policy_digest=graph_policy,
        query_digest=scope.query_digest or "unprepared-lexical-only-v1",
        ranking_policy=digest(
            {
                **ranking_policy(scope),
                "requested_document_ids": sorted(
                    str(v) for v in scope.requested_document_ids
                ),
            }
        ),
        vocabulary_digest="vocabulary-disabled-v1",
    )


def ordered_candidates(
    rows,
    channels,
    *,
    query,
    exact_rows,
    scope,
    graph_authority="graph-disabled-v1",
    graph_policy="graph-disabled-v1",
    preserved_ranks=None,
):
    if not controls()["rrf"]:
        return tuple(rows), None
    by_id = {row.pk: row for row in rows}
    ranks = {
        name: tuple(
            (pk, rank) for rank, pk in enumerate(dict.fromkeys(ids), 1) if pk in by_id
        )
        for name, ids in channels.items()
    }
    for name, entries in (preserved_ranks or {}).items():
        ranks[name] = tuple((pk, rank) for pk, rank in entries if pk in by_id)
    if scope is not None:
        ranks["full_text"] = tuple(
            (pk, rank)
            for pk, rank in ranks.get("full_text", ())
            if pk not in scope.fts_invalid_ids
        )
    protected = (
        strong_exact_ids(tuple(row for row in exact_rows if row.pk in by_id), query)
        if controls()["protected_exact"]
        else ()
    )
    limit = min(626, len(by_id))
    if scope is not None and not scope.query_digest:
        scope.query_digest = digest({"version": "unprepared-query-v1", "query": query})
    result = fuse_ranked_channels_v2(
        ranks,
        limit=limit,
        scope_key=fusion_scope_key(
            scope, graph_authority=graph_authority, graph_policy=graph_policy
        )
        if scope is not None
        else "0" * 64,
        protected_exact_ids=protected,
    )
    if scope is not None:
        scope.fusion_v2 = result
    # Non-ORM adapters order rows without minting a scope-bound record.
    return tuple(
        by_id[row.chunk_id] for row in result.candidates
    ), result if scope else None


def snapshot_channels(snapshot):
    return {
        "dense": snapshot.vector_chunk_ids,
        "exact": snapshot.exact_chunk_ids,
        "trigram": snapshot.trigram_chunk_ids,
        "full_text": snapshot.full_text_chunk_ids,
    }


def fuse_validated_graph(snapshot, fused, query, ready_checksum):
    """Called only after the existing V1 authority and complete mapping checks."""
    from .chunk_search_scope import current_scope

    channels = snapshot_channels(snapshot)
    preserved = {}
    for channel, attribute in (
        ("graph_direct", "direct_rank"),
        ("graph_extended", "extended_rank"),
    ):
        candidates = sorted(
            (row for row in fused.candidates if getattr(row, attribute) is not None),
            key=lambda row: getattr(row, attribute),
        )
        preserved[channel] = tuple(
            (row.integer_chunk_pk, getattr(row, attribute)) for row in candidates
        )
    rows, _ = ordered_candidates(
        fused.rerank_candidates,
        channels,
        query=query,
        exact_rows=snapshot.exact_results,
        scope=current_scope(),
        graph_authority=digest(
            {
                "version": "validated-legacy-graph-v1",
                "ready_bundle_checksum": ready_checksum,
            }
        ),
        graph_policy="validated-legacy-graph-rrf60-v1",
        preserved_ranks=preserved,
    )
    return rows
