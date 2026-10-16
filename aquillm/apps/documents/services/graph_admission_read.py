"""Source-only eager staging for the native plan, including ordering and cleanup."""


def read_owned_refs(plan, refs, reader):
    """Narrow only the original native bound reader in this exact plan scope."""
    from types import MethodType

    from apps.documents.models import TextChunk
    from apps.knowledge_graph.retrieval.graph_candidate_stage_v2 import (
        _NATIVE_GRAPH_SOURCE_READ,
        _GraphSourceStage,
    )

    from .graph_candidate_metadata import captured_query

    if (type(reader) is MethodType
            and type(reader.__self__) is _GraphSourceStage
            and reader.__self__.scope is plan.scope
            and reader.__func__ is _NATIVE_GRAPH_SOURCE_READ):
        return reader.__self__.read_captured(refs)
    return reader(captured_query(TextChunk, refs, plan.scope))


def _read_stage(plan, ids, stage, *, baseline=(), order=False):
    from .chunk_search_scoped_rows import _retain_dense

    plan.check()
    stage.check()
    refs = plan.references
    loaded = stage.read_captured(tuple(refs[pk] for pk in ids)) if ids else ()
    plan.populate(loaded, stage.staged)
    merged = tuple({r.pk: r for r in (*baseline, *loaded)}.values())
    rows, _ = stage.current(merged, stage.initial_keys)
    final_keys = stage.initial_keys & set(plan.scope.validate())
    rows, valid_fts = stage.current(rows, final_keys)
    rows = _retain_dense(rows, stage.staged, final_keys, valid_fts)
    stage.final_keys = frozenset(final_keys)
    return plan.order(rows, scope=stage.staged) if order else rows


def _read_initial_captured_refs(plan, stage):
    """Interior only: the encompassing metadata owner publishes after cleanup."""
    from dataclasses import replace
    from .graph_candidate_metadata import _reconcile_metadata

    plan.check()
    stage.check()
    with plan.scope.database():
        rows = _read_stage(plan, plan.initial_ids, stage)
        channels, dense = _reconcile_metadata(dict(plan.channels), dict(plan.dense), stage.final_keys)
        retained = {ref.pk for refs in channels.values() for ref in refs}
        plan = replace(plan, channels=tuple(channels.items()), dense=tuple(dense.items()),
            initial_ids=tuple(pk for pk in plan.initial_ids if pk in retained))
        ordered = plan.order(rows, scope=stage.staged)
        stage.check()
        plan.check()
    stage.check()
    plan.check()
    return tuple(ordered), plan

def read_captured_refs(plan, ids, *, baseline=(), order=False):
    from apps.documents.models import TextChunk
    from apps.knowledge_graph.retrieval.graph_candidate_stage_v2 import (
        _GraphSourceStage,
    )

    from .graph_candidate_metadata import captured_query

    plan.check()
    scope, refs = plan.scope, plan.references
    query = captured_query(TextChunk, tuple(refs[pk] for pk in ids), scope)
    stage = _GraphSourceStage(scope, query, plan.header)

    def check():
        plan.check()
        stage.check()

    try:
        with plan.header[0][5]:
            check()
            with scope.database():
                check()
                stage.initial_keys = set(scope.validate())
                ordered = _read_stage(plan, ids, stage, baseline=baseline, order=order)
                check()
            # Database cleanup can revoke lifetime or replace captured state.
            check()
            stage.publish(ordered, check)
            plan.check()
            return tuple(ordered)
    except Exception:
        # Charges remain cumulative; no body/cache/fusion is published before
        # the outer cleanup and the final original identity/lifetime gates.
        raise
    finally:
        stage.withdraw()
