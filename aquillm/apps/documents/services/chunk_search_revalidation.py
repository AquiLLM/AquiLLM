"""Withdraw evidence votes without recapturing authority or expanding the pool."""

from .candidate_rank_fusion import CHANNELS, fuse_ranked_channels_v2
from .chunk_search_scope import current_scope


def restrict_fusion(fusion, *, allowed=None, channel_ids=None):
    if fusion is None:
        return None
    channels = {name: [] for name in CHANNELS}
    protected = []
    for row in fusion.candidates:
        if allowed is not None and row.chunk_id not in allowed:
            continue
        for rank in row.channel_ranks:
            if channel_ids is None or row.chunk_id in channel_ids.get(
                rank.channel, {row.chunk_id}
            ):
                channels[rank.channel].append((row.chunk_id, rank.rank))
                if row.protected_exact and rank.channel == "exact":
                    protected.append(row.chunk_id)
    return fuse_ranked_channels_v2(
        {name: tuple(entries) for name, entries in channels.items()},
        limit=len(fusion.candidates),
        scope_key=fusion.scope_key,
        protected_exact_ids=tuple(protected),
    )


def withdraw_full_text(scope, checked_rows, valid_ids):
    rejected = {row.pk for row in checked_rows if row.pk in scope.fts_rows} - valid_ids
    scope.fts_invalid_ids.update(rejected)
    if rejected:
        scope.fusion_v2 = restrict_fusion(
            scope.fusion_v2,
            channel_ids={"full_text": set(scope.fts_rows) - scope.fts_invalid_ids},
        )


def reconcile_candidate_order(rows):
    """Only reorder already admitted rows, before scoring or known fallback."""
    scope = current_scope()
    rows = tuple(rows)
    if scope is None or scope.fusion_v2 is None:
        return rows
    by_id = {row.pk: row for row in rows}
    ranked = tuple(
        by_id[c.chunk_id] for c in scope.fusion_v2.candidates if c.chunk_id in by_id
    )
    known = {row.pk for row in ranked}
    return ranked + tuple(row for row in rows if row.pk not in known)


def ranking_status(reranked, score_set):
    status = (
        score_set.status
        if score_set is not None
        else getattr(reranked, "_rag_rerank_status", "unknown")
    )
    return (
        status
        if type(status) is str and status in ("complete", "unavailable")
        else "unknown"
    )


def final_fusion(scope, snapshot, pool, *, full_text_ids, dense_ids):
    if scope is None:
        return snapshot.fusion_v2
    scope.fusion_v2 = restrict_fusion(
        scope.fusion_v2,
        allowed={row.pk for row in pool},
        channel_ids={"full_text": full_text_ids, "dense": dense_ids},
    )
    return scope.fusion_v2
