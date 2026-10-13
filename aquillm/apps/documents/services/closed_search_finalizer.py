"""Private native row authorizer and staged post-inference ranking closure."""

from dataclasses import dataclass

from .chunk_search_authorization import authorized_search_rows as _native_authorize
from .chunk_search_scope import current_scope
from .chunk_search_source_operation import (
    _closed_header,
    _closed_row_operation,
    _same_header,
)
from .representation_contracts import SourceScopeUnavailable
from .source_loading import bounded_source_database


@dataclass(frozen=True, slots=True)
class _NativeSearchAuthorizer:
    delegate: object
    authorization: object
    hybrid_requested: bool
    scope: object
    queryset: object
    header: object

    def __call__(self, rows):
        return self.delegate(
            rows,
            authorization_context=self.authorization,
            hybrid_requested=self.hybrid_requested,
        )


def _bind_search_authorizer(delegate, authorization, hybrid_requested):
    from apps.documents.models import TextChunk

    from .chunk_search_rank_policy import controls
    from .chunk_search_scope import FrozenRetrievalScope

    scope, query, header = current_scope(), None, None
    if (
        delegate is _native_authorize
        and not controls()["full_text"]
        and type(scope) is FrozenRetrievalScope
    ):
        query = TextChunk.objects.using(scope.using).all()
        header = _closed_header(query, scope)
    return _NativeSearchAuthorizer(
        delegate, authorization, hybrid_requested, scope, query, header
    )


def _finalize_closed_ranking(snapshot, ranking, *, authorize, query, top_k):
    from . import chunk_search_scoped_rows as row_operations
    from .chunk_rerank_results import rank_only_result_for_chunks
    from .chunk_search_authorization import _finish_search_rows, _incoming_search_rows
    from .chunk_search_rank_policy import controls
    from .chunk_search_score_handoff import score_set_for_authorized_rows

    if (
        type(authorize) is not _NativeSearchAuthorizer
        or authorize.delegate is not _native_authorize
        or authorize.header is None
    ):
        return None
    scope, header, base = authorize.scope, authorize.header, authorize.queryset
    if (
        controls()["full_text"]
        or current_scope() is not scope
        or not _same_header(header, _closed_header(base, scope))
    ):
        raise SourceScopeUnavailable("source_scope_unavailable")
    groups = (
        tuple(snapshot.vector_results),
        tuple(snapshot.trigram_results),
        tuple(ranking.ranked_results),
        tuple(snapshot.full_text_results),
        tuple(snapshot.exact_results),
        tuple(ranking.combined_candidates),
    )
    status = (
        ranking.score_set.status
        if ranking.score_set is not None
        else ranking.rerank_status
    )

    def check():
        if current_scope() is not scope or controls()["full_text"]:
            raise SourceScopeUnavailable("source_scope_unavailable")

    def read(staged, original, budget, allowed, cache, initial_keys):
        rows = tuple(row for group in groups for row in group)
        rows, _ = row_operations._current_rows(rows, staged, initial_keys)
        return _incoming_search_rows(
            rows,
            authorization_context=authorize.authorization,
            hybrid_requested=authorize.hybrid_requested,
        )

    def finish(rows, staged, keys, valid_fts):
        final, channels = _finish_search_rows(
            snapshot,
            groups,
            {id(row) for row in rows},
            staged,
            keys,
            valid_fts,
            snapshot.full_text_coverage,
            top_k if status != "complete" else None,
        )
        scores = ranking.score_set
        if scores is not None:
            scores = (
                score_set_for_authorized_rows(scores, tuple(channels[2]))
                if status == "complete"
                else rank_only_result_for_chunks(
                    query, tuple(channels[2]), top_k
                ).score_set
            )
        return final, channels, scores

    with header[0][5], bounded_source_database(header[0][1], header[1][0]):
        if not _same_header(header, _closed_header(base, scope)):
            raise SourceScopeUnavailable("source_scope_unavailable")
        return _closed_row_operation(
            base, scope, header, read, finish=finish, check=check
        )
