"""Source-only eager staging for the native plan, including ordering and cleanup."""

def read_captured_refs(plan, ids, *, baseline=(), order=False):
    from apps.documents.models import TextChunk
    from apps.knowledge_graph.retrieval.graph_candidate_stage_v2 import (
        _GraphSourceStage,
    )

    from .chunk_search_scoped_rows import _retain_dense
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
                loaded = (
                    stage.read_captured(tuple(refs[pk] for pk in ids)) if ids else ()
                )
                plan.populate(loaded, stage.staged)
                merged = tuple({r.pk: r for r in (*baseline, *loaded)}.values())
                rows, _ = stage.current(merged, stage.initial_keys)
                final_keys = stage.initial_keys & set(scope.validate())
                rows, valid_fts = stage.current(rows, final_keys)
                rows = _retain_dense(rows, stage.staged, final_keys, valid_fts)
                ordered = plan.order(rows, scope=stage.staged) if order else rows
                check()
            # Database cleanup can revoke lifetime or replace captured state.
            check()
            stage.final_keys = frozenset(final_keys)
            stage.publish(ordered, check)
            plan.check()
            return tuple(ordered)
    except Exception:
        # Charges remain cumulative; no body/cache/fusion is published before
        # the outer cleanup and the final original identity/lifetime gates.
        raise
    finally:
        stage.withdraw()
