"""Small private seams for the closed V2 owner; no public bypass switches."""

from .graph_source_admission import NativeSourceAdmission
from .representation_contracts import SourceScopeUnavailable


def current_lexical_ids(scope):
    plan = getattr(scope, "named_item_admission", None)
    if plan is None:
        return scope.lexical_ids
    from .chunk_search_scope import current_scope

    original = current_scope()
    if (type(plan) is not NativeSourceAdmission or plan.scope is not original
            or original is None or scope.authorization is not original.authorization
            or scope.runtime is not original.runtime or scope.representation is not original.representation
            or scope.requested_document_ids != original.requested_document_ids
            or scope.query_digest != original.query_digest
            or (original.named_item_admission is not None and original.named_item_admission is not plan)):
        raise SourceScopeUnavailable("named_item_scope_unavailable")
    return scope.lexical_ids.union(plan.channel_ids.get("named_items", ()))


def captured_admission(snapshot, scope, *, query=None):
    plan = getattr(snapshot, "source_admission", None)
    if plan is not None:
        if (type(plan) is not NativeSourceAdmission or plan.scope is not scope
                or (query is not None and plan.query != query)):
            raise SourceScopeUnavailable("source_admission_original_scope_unavailable")
        plan.check()
    return plan


def selector_body_reader(plan, stage):
    if plan is None:
        return stage.read
    return lambda query: plan.graph_read(query, stage.read)


def deferred_baseline(plan, stage, baseline):
    if plan is None:
        return baseline
    added = plan.top_up(stage.read, stage.staged)
    return _merge_baseline(plan, baseline, added)


def _merge_baseline(plan, baseline, added):
    by_id = {row.pk: row for row in (*baseline, *added)}
    return tuple(by_id[pk] for pk in plan.baseline_order if pk in by_id)


def native_deferred_baseline(plan, stage, baseline, check):
    """Let a deferred baseline read expire before the graph's mandatory final fences."""
    from django.db import OperationalError

    from .graph_admission_read import read_owned_refs
    from .native_search_schedule import NATIVE_CLOSURE_RESERVE_MS
    from .source_deadline import _DEADLINE, source_preparation_scope
    from .source_loading import SourcePreparationLimited, bounded_source_database

    deadline, clock, budget = _DEADLINE.get()
    cutoff = deadline - NATIVE_CLOSURE_RESERVE_MS / 1000
    added = ()
    if (cutoff - clock()) * 1000 >= 1:
        try:
            with stage.buffered_reads(), source_preparation_scope(cutoff, clock, budget):
                ids = plan.top_up_ids()
                if ids:
                    refs = plan.references
                    with bounded_source_database(stage.runtime, stage.scope.using):
                        added = read_owned_refs(plan, tuple(refs[pk] for pk in ids), stage.read)
        except (SourcePreparationLimited, OperationalError) as exc:
            # A real owner failure or an unrelated SQL error must still fail closed.
            check()
            if (cutoff - clock()) * 1000 >= 1:
                raise
            if isinstance(exc, OperationalError):
                cause = exc.__cause__
                code = getattr(cause, "sqlstate", None) or getattr(cause, "pgcode", None)
                message = getattr(getattr(cause, "diag", None), "message_primary", None)
                if code != "57014" or message != "canceling statement due to statement timeout":
                    raise
            elif exc.args not in (("source preparation deadline",), ("source deadline",)):
                raise
            added = ()
    check()
    if added:
        plan.populate(added, stage.staged)
    plan.counts["baseline_top_up"] = len(added)
    return _merge_baseline(plan, baseline, added)


def order_graph_pool(snapshot, rows, *, scope, query, **kwargs):
    plan = getattr(snapshot, "source_admission", None)
    if plan is not None:
        plan.check()
        return plan.order(rows, scope=scope, **kwargs)
    from .chunk_search_rank_policy import ordered_candidates, snapshot_channels

    return ordered_candidates(
        rows, snapshot_channels(snapshot), query=query,
        exact_rows=snapshot.exact_results, scope=scope, **kwargs,
    )[0]


def fallback_baseline(snapshot, *, admit_deferred=True):
    plan = getattr(snapshot, "source_admission", None)
    if plan is None:
        return snapshot.baseline_candidates
    ids = plan.top_up_ids() if admit_deferred else ()
    rows = plan._closed_read(ids, baseline=snapshot.baseline_candidates, order=True)
    plan.counts["fallback_top_up_skipped"] = not admit_deferred
    plan.counts["fallback_top_up_count"] = len({r.pk for r in rows} - set(plan.initial_ids))
    return rows


def observe_published_graph(pks):
    try:
        from lib.replay_observation import graph_selected_sources_v2

        graph_selected_sources_v2(pks)
    except Exception:
        pass  # Best-effort provenance cannot replace a successful native result.
