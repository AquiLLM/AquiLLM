"""Private staged eager source/row operation with fresh scope boundaries."""

from collections import ChainMap
from copy import deepcopy
from dataclasses import replace

from .chunk_search_scope import FrozenRetrievalScope
from .representation_contracts import SourceScopeUnavailable
from .source_loading import SourceRuntime, _prepare_source_queryset, _source_body_rows

_OBSERVATIONS = ("bodies", "dense_rows", "lexical_ids", "fts_rows", "fts_invalid_ids")


def _closed_header(queryset, scope):
    """Capture identity only inside the eager operation; never an auth permit."""
    from apps.documents.models import TextChunk, TextChunkQuerySet
    from lib.retrieval.turn_budget import TurnBudget

    from .candidate_rank_fusion import FusionResultV2
    from .hybrid_graph_authorization import is_exact_authorization_context

    runtime = scope.runtime
    original = getattr(runtime, "_frozen_retrieval_scope", None)
    if (
        type(queryset) is not TextChunkQuerySet
        or queryset.model is not TextChunk
        or type(original) is not FrozenRetrievalScope
        or type(runtime) is not SourceRuntime
        or type(runtime.budget) is not TurnBudget
        or type(runtime.cache) is not dict
        or not is_exact_authorization_context(runtime.authorization)
        or original.runtime is not runtime
        or original.authorization is not runtime.authorization
        or scope.authorization is not runtime.authorization
        or original.proposal_binding is not None
        or original.representation is not scope.representation
        or original.dense is not scope.dense
        or original.space is not scope.space
        or not queryset.db
        == scope.using
        == original.using
        == runtime.authorization.database_alias
        or type(scope.requested_document_ids) is not frozenset
        or original.requested_document_ids
        != frozenset(
            selection.document.id for selection in scope.representation.selections
        )
        or not scope.requested_document_ids <= original.requested_document_ids
        or any(
            type(getattr(scope, name)) is not expected
            for name, expected in zip(
                _OBSERVATIONS, (dict, dict, set, dict, set), strict=True
            )
        )
        or (scope.fusion_v2 is not None and type(scope.fusion_v2) is not FusionResultV2)
    ):
        return None
    return (
        (
            original,
            runtime,
            runtime.authorization,
            runtime.budget,
            runtime.cache,
            runtime.lock,
            scope.representation,
            scope.dense,
            scope.space,
            scope.proposal_binding,
        ),
        (
            queryset.db,
            scope.using,
            original.using,
            scope.requested_document_ids,
            original.requested_document_ids,
            scope.query_digest,
            scope.query_compatibility,
            scope.vocabulary_digest,
            scope.graph_enabled,
        ),
    )


def _same_header(left, right):
    return (
        right is not None
        and all(a is b for a, b in zip(left[0], right[0], strict=True))
        and left[1] == right[1]
    )


def _closed_scoped_rows(queryset, scope, header, lexical):
    from . import chunk_search_scoped_rows as row_operations

    def read(staged, original, budget, allowed, cache):
        query = _prepare_source_queryset(queryset, original)
        rows = (
            _source_body_rows(query, original, budget, allowed, cache)
            if allowed and budget.can_publish()
            else ()
        )
        row_operations._record_rows(queryset, rows, staged, lexical)
        return rows

    return _closed_row_operation(
        queryset, scope, header, read, full_text=lexical == "full_text"
    )


def _closed_row_operation(queryset, scope, header, read, *, full_text=False):
    """Consume a private eager reader; shared fresh fences own all publication."""
    from apps.collections.services.retrieval_authorization import (
        revalidate_retrieval_authorization_context,
    )

    from . import chunk_search_scoped_rows as row_operations
    from .chunk_search_revalidation import restrict_fusion
    from .source_deadline import check_source_deadline

    original, runtime = header[0][:2]
    budget = header[0][3]
    containers = {name: getattr(scope, name) for name in _OBSERVATIONS}
    before = {name: deepcopy(value) for name, value in containers.items()}
    fusion = scope.fusion_v2
    staged = replace(scope, **deepcopy(before))
    pending = {}
    try:
        initial_keys = set(scope.validate())
        authority = revalidate_retrieval_authorization_context(
            context=runtime.authorization
        )
        allowed = frozenset(authority.document_ids)
        rows = tuple(
            read(staged, original, budget, allowed, ChainMap(pending, header[0][4]))
        )
        rows, valid_fts = row_operations._current_rows(rows, staged, initial_keys)
        final_keys = initial_keys & set(scope.validate())
        rows = row_operations._retain_dense(rows, staged, final_keys, valid_fts)
        if full_text:
            rows = tuple(row for row in rows if row.pk not in staged.fts_invalid_ids)
        check_source_deadline()
        surviving = {row.pk for row in rows}
        identities = {(row.pk, row_operations.fingerprint(row)) for row in rows}

        def publish():
            # Check inside the ledger gate too: its clock/liveness callbacks may
            # reenter this RLock before invoking the publication callback.
            check_source_deadline()
            if (
                not _same_header(header, _closed_header(queryset, scope))
                or scope.fusion_v2 is not fusion
                or any(
                    getattr(scope, name) is not containers[name]
                    or getattr(scope, name) != before[name]
                    for name in _OBSERVATIONS
                )
            ):
                raise SourceScopeUnavailable("source_scope_unavailable")
            for pk in surviving:
                scope.bodies.setdefault(pk, staged.bodies[pk])
                if pk in staged.lexical_ids:
                    scope.lexical_ids.add(pk)
                if pk in staged.fts_rows:
                    scope.fts_rows.setdefault(pk, staged.fts_rows[pk])
            header[0][4].update(
                {
                    key: row
                    for key, row in pending.items()
                    if (row.pk, row_operations.fingerprint(row)) in identities
                }
            )

        if not budget.publish(publish):
            raise SourceScopeUnavailable("source_scope_unavailable")
        return rows
    finally:
        # Negative evidence is sticky even if the final policy/deadline aborts.
        # Old integer PKs must never prune a replacement semantic scope's fusion.
        rejected = staged.fts_invalid_ids - before["fts_invalid_ids"]
        containers["fts_invalid_ids"].update(rejected)
        if (
            rejected
            and _same_header(header, _closed_header(queryset, scope))
            and scope.fts_rows is containers["fts_rows"]
            and scope.fts_invalid_ids is containers["fts_invalid_ids"]
        ):
            scope.fusion_v2 = restrict_fusion(
                scope.fusion_v2,
                channel_ids={"full_text": set(scope.fts_rows) - scope.fts_invalid_ids},
            )
