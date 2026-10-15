"""Focused native closed-seed policy contracts."""

import pytest
from django.db import connection

from apps.collections.services.bounded_retrieval_authorization import (
    bounded_reauthorization_reads,
)
from apps.documents.models import RawTextDocument
from apps.knowledge_graph.retrieval.extended_seed_repository import (
    ExtendedSeedRepository,
)

from .native_closed_seed_policy_fixtures import (
    native_frame,
    policy_data_counts,
    prepare,
)

pytestmark = pytest.mark.django_db(transaction=True, databases="__all__")


def test_real_31_530_ordinary_batch_policy_baseline_is_four_statements_1122_rows(
    monkeypatch,
):
    from apps.knowledge_graph.retrieval.extended_seed_repository import (
        ExtendedSeedRepository,
    )

    with native_frame(monkeypatch, large=True) as frame:
        active, owner, view, groups, budget = frame
        _, graph, _, scope, _, _, _, _ = active
        requests = tuple(
            (
                p,
                tuple(
                    (r.pk, r.doc_id)
                    for r in view.seed_references
                    if r.doc_id in dict(p.documents)
                ),
            )
            for group in groups
            for p in group.scope.projections
        )
        requests = tuple((p, chunks) for p, chunks in requests if chunks)
        with (
            bounded_reauthorization_reads(scope.authorization, budget),
            policy_data_counts() as counts,
        ):
            mapping = ExtendedSeedRepository()._load_seed_identities_batch(
                requests=requests,
                authorization=scope.authorization,
                codec=graph.codec,
                max_rows=4999,
                read_budget=budget,
            )
        assert mapping and all(any(value.values()) for value in mapping.values())
        assert len(scope.authorization.selected_collection_ids) == 31
        assert len(scope.authorization.selected_document_ids) == 530
        assert (len(counts), sum(counts)) == (4, 1122)
        print(
            f"ordinary native policy boundaries: statements={len(counts)}, "
            f"returned_rows={sum(counts)}"
        )


def test_real_31_530_native_owner_uses_two_fresh_single_row_policy_proofs(monkeypatch):
    with native_frame(monkeypatch, large=True) as frame:
        active, owner, _, _, budget = frame
        _, _, source, scope, _, _, _, _ = active
        original = (
            source.budget,
            scope.authorization,
            owner.plan,
            budget.deadline,
            budget.clock,
        )
        before = (source.budget.sources_used, dict(source.cache), dict(scope.bodies))
        with policy_data_counts() as counts:
            prepare(frame)
        assert owner.mapping and all(
            any(values.values()) for values in owner.mapping.values()
        )
        assert budget.turn_budget is original[0] and scope.authorization is original[1]
        assert (
            owner.plan is original[2]
            and budget.deadline == original[3]
            and budget.clock is original[4]
        )
        assert before == (source.budget.sources_used, source.cache, scope.bodies)
        assert (budget.max_queries, budget.max_source_rows, budget.max_wire_bytes) == (
            64,
            5000,
            1048576,
        )
        print(
            f"closed native policy boundaries: statements={len(counts)}, "
            f"returned_rows={sum(counts)}; "
            f"original frontier statements={budget.queries}, "
            f"returned_rows={budget.source_rows}"
        )
        assert (len(counts), sum(counts)) == (2, 2), (
            "native full-scope proof still materializes selected policy rows"
        )


def test_document_input_is_globally_capped_at_n_plus_one_before_counts(monkeypatch):
    with native_frame(monkeypatch) as frame:
        a = frame[0][0].documents["A"]
        RawTextDocument.objects.bulk_create(
            [
                RawTextDocument(
                    id=a.id,
                    title=f"overflow-{i}",
                    full_text=f"overflow-{i}",
                    full_text_hash=str(i) * 64,
                    collection_id=a.collection_id,
                    ingested_by_id=a.ingested_by_id,
                )
                for i in range(1, 7)
            ]
        )
        authorization = frame[0][3].authorization
        with bounded_reauthorization_reads(authorization, frame[4]):
            proof = frame[1]._native_policy.select(
                authorization=authorization,
                repository=ExtendedSeedRepository(),
                read_budget=frame[4],
            )
            sql, params = proof._statement()
            with connection.cursor() as cursor:
                cursor.execute(sql, params)
                rows = cursor.fetchall()
            assert len(rows) == 1 and rows[0][2] == 3
            with pytest.raises(ValueError):
                proof.require_current()
        assert (frame[4].queries, frame[4].source_rows) == (1, 1)


def test_actual_postgres_plan_bounds_undeduplicated_inputs_before_count(monkeypatch):
    with native_frame(monkeypatch, large=True) as frame:
        active, owner, _, _, budget = frame
        with bounded_reauthorization_reads(active[3].authorization, budget):
            proof = owner._native_policy.select(
                authorization=active[3].authorization,
                repository=ExtendedSeedRepository(),
                read_budget=budget,
            )
            sql, params = proof._statement()
            with connection.cursor() as cursor:
                cursor.execute("EXPLAIN (ANALYZE, FORMAT JSON) " + sql, params)
                plan = cursor.fetchone()[0][0]["Plan"]

        def walk(node):
            yield node
            for child in node.get("Plans", ()):
                yield from walk(child)

        nodes = tuple(walk(plan))
        permissions = next(
            n
            for n in nodes
            if n.get("Subplan Name") == "CTE native_seed_policy_permissions"
        )
        documents = next(
            n
            for n in nodes
            if n.get("Subplan Name") == "CTE native_seed_policy_documents"
        )
        assert permissions["Node Type"] == documents["Node Type"] == "Limit"
        assert permissions["Actual Rows"] == 31 and documents["Actual Rows"] == 530
        assert documents["Plans"][0]["Node Type"] == "Append"
        assert len(documents["Plans"][0]["Plans"]) == 8
        # A permission semijoin may hash its permission input. The complete
        # document relation itself remains an undeduplicated Append below Limit.
        # Keep statement parameters, source bodies and filter expressions private.
        print(
            "native policy bounded plan:",
            [
                (n["Node Type"], n.get("Actual Rows"), n.get("Index Name"))
                for n in nodes
            ],
        )
