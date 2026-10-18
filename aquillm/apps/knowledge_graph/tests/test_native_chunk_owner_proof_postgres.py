"""Actual closed projection batch keeps policy/projection/identity order."""

import pytest
from django.db import connections

from apps.collections.tests.native_closed_seed_policy_fixtures import (
    native_frame,
    prepare,
)
from apps.knowledge_graph.models import (
    CollectionEntityDocumentLink,
    CollectionGraphProjection,
)
from apps.knowledge_graph.retrieval.graph_runtime_seeds_v2 import _group_request_native

pytestmark = pytest.mark.django_db(transaction=True, databases="__all__")


def bind(frame):
    active, owner, _, groups, budget = frame
    _group_request_native(
        owner=owner,
        query="Atlas",
        group=groups[0],
        authorization=active[3].authorization,
        read_budget=budget,
    )


def test_actual_native_batch_uses_chunk_proof_and_keeps_both_barriers(monkeypatch):
    with native_frame(monkeypatch, metadata=True) as frame:
        bind(frame)
        events = []
        budget = frame[4]
        before = (budget.queries, budget.source_rows)

        def observe(execute, sql, params, many, context):
            result = execute(sql, params, many, context)
            if sql.startswith("WITH native_seed_policy"):
                events.append(("policy", context["cursor"].rowcount))
            elif sql.startswith("WITH expected(id,doc_id)"):
                events.append(("chunk", context["cursor"].rowcount))
            elif CollectionGraphProjection._meta.db_table in sql and sql.startswith(
                "SELECT"
            ):
                events.append(("projection", context["cursor"].rowcount))
            elif CollectionEntityDocumentLink._meta.db_table in sql:
                events.append(("identity", context["cursor"].rowcount))
            return result

        with (
            connections["default"].execute_wrapper(observe),
            connections["projection_source"].execute_wrapper(observe),
        ):
            prepare(frame)
        assert [e[0] for e in events] == [
            "policy",
            "projection",
            "chunk",
            "identity",
            "policy",
            "projection",
        ]
        assert events[0][1] == events[2][1] == events[4][1] == 1
        assert events[1][1] == events[5][1]
        assert events[3][1] > 1
        assert (budget.queries - before[0], budget.source_rows - before[1]) == (
            6,
            sum(e[1] for e in events),
        )
        assert frame[1].mapping and not frame[1]._native_metadata.failed
        assert not budget._uncertain and budget._reserved_rows == 0


@pytest.mark.parametrize("change", ["chunk", "permission", "projection"])
def test_native_batch_stale_owner_or_retained_final_guard_rejects(monkeypatch, change):
    from apps.collections.models import CollectionPermission
    from apps.documents.models import TextChunk

    with native_frame(monkeypatch, metadata=True) as frame:
        bind(frame)
        active, owner, view, _, budget = frame
        if change == "chunk":
            # Real current ownership changes; signed expected references
            #  remain unchanged.
            def move_after_initial_projection(execute, sql, params, many, context):
                result = execute(sql, params, many, context)
                if CollectionGraphProjection._meta.db_table in sql and sql.startswith(
                    "SELECT"
                ):
                    TextChunk.objects.filter(pk=view.seed_references[0].pk).update(
                        doc_id=active[0].documents["B"].id
                    )
                return result

            with (
                connections["projection_source"].execute_wrapper(
                    move_after_initial_projection
                ),
                pytest.raises(ValueError, match="chunk ownership"),
            ):
                prepare(frame)
            assert owner._native_metadata.failed
        else:

            def change_after_identity(execute, sql, params, many, context):
                result = execute(sql, params, many, context)
                if CollectionEntityDocumentLink._meta.db_table in sql:
                    if change == "permission":
                        CollectionPermission.objects.filter(collection_id=7).delete()
                    else:
                        p = next(
                            p
                            for p in owner.projections
                            if any(
                                r.doc_id in dict(p.documents)
                                for r in view.seed_references
                            )
                        )
                        CollectionGraphProjection.objects.filter(
                            pk=p.projection_id
                        ).update(membership_epoch=p.membership_epoch + 1)
                return result

            with (
                connections["projection_source"].execute_wrapper(change_after_identity),
                pytest.raises(ValueError),
            ):
                prepare(frame)
        assert budget.queries > 2
