"""Small private seams for the closed V2 owner; no public bypass switches."""

from .graph_source_admission import NativeSourceAdmission
from .representation_contracts import SourceScopeUnavailable


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
    by_id = {row.pk: row for row in (*baseline, *added)}
    return tuple(by_id[pk] for pk in plan.baseline_order if pk in by_id)


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


def fallback_baseline(snapshot):
    plan = getattr(snapshot, "source_admission", None)
    if plan is None:
        return snapshot.baseline_candidates
    return plan._closed_read(plan.top_up_ids(), baseline=snapshot.baseline_candidates, order=True)


def observe_published_graph(pks):
    try:
        from lib.replay_observation import graph_selected_sources_v2

        graph_selected_sources_v2(pks)
    except Exception:
        pass  # Best-effort provenance cannot replace a successful native result.
