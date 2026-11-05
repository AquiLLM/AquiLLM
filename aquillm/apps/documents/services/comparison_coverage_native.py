"""Optional paired-paper acquisition inside the original native source owner."""

from dataclasses import replace
from time import perf_counter
from types import SimpleNamespace
from uuid import UUID

from .comparison_coverage_policy import (
    CoverageGroup,
    bind_coverage,
    capture_source,
    comparison_coverage_enabled,
    mixed_dense,
    mixed_lexical,
    native_successor,
    stored_overlap,
)
from .comparison_version_query import parse_version_comparison, resolve_primary_documents
from .representation_contracts import SourceScopeUnavailable


def _scoped_metadata(model, query, scope, limits, observation, stage, check, documents):
    from django.db import OperationalError
    from apps.chat.services.rag_acquisition import ACTION_TIMEOUT_MS, COMPLETION_RESERVE_MS
    from lib.retrieval.operation import _OPERATION

    from .graph_candidate_metadata import _capture_metadata
    from .source_deadline import _DEADLINE, check_source_deadline, source_preparation_scope, source_remaining_ms
    from .source_loading import SourcePreparationLimited

    budget = scope.runtime.budget
    results = []
    for document in documents:
        document_id = UUID(document)
        if document_id not in scope.requested_document_ids:
            raise SourceScopeUnavailable("comparison_document_scope_changed")
        if not budget.can_start_optional(ACTION_TIMEOUT_MS, COMPLETION_RESERVE_MS):
            return None
        if (not budget.reserve_action("comparison-metadata:" + document)
                or not budget.reserve_text(len(query), kind="materialized")):
            return None
        check()
        stage.check()
        parent = _DEADLINE.get()
        clock = parent[1] if parent is not None else _OPERATION.get().clock
        milliseconds = min(ACTION_TIMEOUT_MS, source_remaining_ms(budget.remaining_ms()) - COMPLETION_RESERVE_MS)
        if milliseconds <= 0:
            return None
        narrowed = replace(scope, requested_document_ids=frozenset({document_id}))

        def child_check():
            check_source_deadline()
            stage.check()

        deadline = clock() + milliseconds / 1000
        try:
            with source_preparation_scope(deadline, clock, budget), scope.database():
                channels, dense, _, _, _, error = _capture_metadata(
                    model, query, narrowed, limits, observation, child_check,
                    stage.initial_keys, None, None, None)
                if error is not None or not channels["dense"] or not channels["trigram"]:
                    return None
                if any(ref.doc_id != document_id for lane in channels.values() for ref in lane):
                    raise SourceScopeUnavailable("comparison_document_scope_changed")
                results.append((channels, dense))
        except (SourcePreparationLimited, TimeoutError, OperationalError) as exc:
            # Only the expired optional child may return the captured global
            # lanes. Cleanup and the original owner checks precede that decision.
            check()
            stage.check()
            if isinstance(exc, OperationalError):
                cause = exc.__cause__
                code = getattr(cause, "pgcode", None) or getattr(cause, "sqlstate", None)
                if code != "57014":
                    raise
            if (deadline - clock()) * 1000 >= 1:
                raise
            with scope.database():
                if set(scope.validate()) != set(stage.initial_keys):
                    raise SourceScopeUnavailable("comparison_source_scope_changed") from exc
            check()
            stage.check()
            return None
        # The owner checks its original deadline object after child cleanup.
        check()
        stage.check()
    return tuple(results)


def _merge_lanes(channels, dense, scoped, keys):
    from .graph_candidate_metadata import _reconcile_metadata

    refs = {name: {ref.pk: ref for ref in lane} for name, lane in channels.items()}
    all_dense = dict(dense)
    for local, vectors in scoped:
        for name, lane in local.items():
            for ref in lane:
                previous = refs[name].get(ref.pk)
                if previous is not None and (previous.fingerprint != ref.fingerprint or previous.length != ref.length):
                    raise SourceScopeUnavailable("comparison_metadata_revision_changed")
                refs[name].setdefault(ref.pk, ref)
        for pk, vector in vectors.items():
            if pk in all_dense and all_dense[pk] != vector:
                raise SourceScopeUnavailable("comparison_dense_revision_changed")
            all_dense[pk] = vector
    _reconcile_metadata({name: tuple(lane.values()) for name, lane in refs.items()}, all_dense, keys)
    heads = tuple(local["trigram"][0].pk for local, _ in scoped)
    lexical = mixed_lexical(tuple(ref.pk for ref in channels["trigram"]), heads)
    if lexical is None:
        return None
    vectors = mixed_dense(tuple(ref.pk for ref in channels["dense"]),
        *(tuple(ref.pk for ref in local["dense"]) for local, _ in scoped))
    updated = {**channels, "dense": tuple(refs["dense"][pk] for pk in vectors),
               "trigram": tuple(refs["trigram"][pk] for pk in lexical)}
    updated, selected_dense = _reconcile_metadata(updated, {pk: all_dense[pk] for pk in vectors}, keys)
    return updated, selected_dense, heads


def _read_groups(scope, stage, channels, heads, check):
    refs = {ref.pk: ref for lane in channels.values() for ref in lane}
    selections = {selection.document.id: selection for selection in scope.representation.selections}
    plans = []
    for pk in heads:
        head = refs[pk]
        representation = selections.get(head.doc_id)
        if representation is None:
            raise SourceScopeUnavailable("comparison_document_identity_unavailable")
        plans.append((head, native_successor(head, tuple(refs.values()), representation), representation))
    requested = tuple({ref.pk: ref for head, successor, _ in plans
                       for ref in ((head, successor) if successor else (head,))}.values())
    check()
    stage.check()
    loaded = {row.pk: row for row in stage.read_captured(requested)}
    if set(loaded) != {ref.pk for ref in requested}:
        return None
    groups = []
    for head, successor, representation in plans:
        sources = [capture_source(head, loaded[head.pk])]
        overlap = None
        if successor is not None:
            continuation = capture_source(successor, loaded[successor.pk])
            overlap = stored_overlap(head, successor, loaded[head.pk].__dict__["content"],
                                     loaded[successor.pk].__dict__["content"])
            if overlap is not None:
                sources.append(continuation)
        groups.append(CoverageGroup(tuple(sources), representation, overlap))
    check()
    stage.check()
    return tuple(groups)


def prepare_comparison_coverage(model, query, scope, limits, observation, channels, dense, *, stage, check):
    """Return scoped candidates and an unpublished record, or the original lanes."""
    from lib.retrieval.operation import _OPERATION, operation_source_limit

    from .chunk_search_rank_policy import controls
    from .graph_candidate_metadata import _reconcile_metadata

    runtime = scope.runtime
    budget = runtime.budget
    operation = _OPERATION.get()
    if (not comparison_coverage_enabled() or parse_version_comparison(query) is None
            or observation is None or scope.dense is None or runtime.comparison_coverage is not None
            or operation is None or operation.budget is not budget
            or budget.actions_used != 1 or budget.sources_used != 0
            or operation_source_limit(budget) not in (30, 45)
            or (limits.vector, limits.trigram, limits.exact) != (17, 17, 17)
            or controls()["full_text"] or len(channels["dense"]) != 17):
        return channels, dense, None
    started = perf_counter()
    check()
    cue = tuple(channels["dense"][:5])
    rows = {row.pk: row for row in stage.read_captured(cue)}
    if set(rows) != {ref.pk for ref in cue}:
        return channels, dense, None
    views = []
    for ref in cue:
        capture_source(ref, rows[ref.pk])
        views.append(SimpleNamespace(chunk_id=ref.pk, document_id=str(ref.doc_id),
                                     text=rows[ref.pk].__dict__["content"]))
    decision = resolve_primary_documents(query, views)
    if len(decision["documents"]) != 2:
        return channels, dense, None
    scoped = _scoped_metadata(model, query, scope, limits, observation, stage, check, decision["documents"])
    if scoped is None:
        return channels, dense, None
    mixed = _merge_lanes(channels, dense, scoped, set(scope.validate()))
    if mixed is None:
        return channels, dense, None
    updated, selected_dense, heads = mixed
    groups = _read_groups(scope, stage, updated, heads, check)
    if groups is None:
        return updated, selected_dense, None
    updated, selected_dense = _reconcile_metadata(updated, selected_dense, set(scope.validate()))
    check()
    stage.check()
    record = bind_coverage(scope, query, groups)
    from lib.evidence_observation import publish
    publish("comparison_coverage_prepared", dict(policy=record.policy, groups=len(groups),
        sources=sum(len(group.sources) for group in groups), metadata_queries=2,
        duration_ms=max(0, (perf_counter() - started) * 1000)))
    return updated, selected_dense, record
