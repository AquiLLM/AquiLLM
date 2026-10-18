"""Product SQL paired31 plans, physical controls and16 chunk proof records."""

from time import monotonic
from types import SimpleNamespace

import pytest
from django.db import connection

from apps.knowledge_graph.retrieval import direct_seed_queries as queries
from apps.knowledge_graph.retrieval.direct_seed_contracts import (
    DirectResolutionTier as Tier,
)
from apps.knowledge_graph.retrieval.direct_seed_queryset import _candidate_queryset
from apps.knowledge_graph.retrieval.direct_seed_sql import bounded_seed_read
from apps.knowledge_graph.retrieval.native_direct_seed_statement import (
    candidate_rows,
    statement,
)
from apps.knowledge_graph.tests.native_metadata_plan_support import (
    find,
    first_limit,
    many,
    subtree,
)
from apps.knowledge_graph.tests.native_metadata_test_support import (
    StatementHarness,
    prepare_tier,
)
from apps.knowledge_graph.tests.test_direct_seed_sql_postgres import (
    scoped_aliases as scoped_aliases,
)

pytestmark = pytest.mark.django_db


@pytest.mark.parametrize("tier", tuple(Tier))
def test_product_paired31_candidate_plan_and_control_reduction(scoped_aliases, tier):
    fixture = many(scoped_aliases)
    options = prepare_tier(fixture, tier)
    original_harness, native = StatementHarness(), StatementHarness()
    counts = []

    def observe(execute, sql, params, many, context):
        counts.append(
            "data"
            if sql.startswith("WITH expected")
            or (
                sql.startswith("SELECT")
                and "set_config(" not in sql
                and "current_setting(" not in sql
            )
            else "control"
        )
        return execute(sql, params, many, context)

    with connection.execute_wrapper(observe):
        states = queries._load_membership_states(
            collection_ids=fixture[0].selected_collection_ids,
            using="default",
            deadline=options["deadline"],
            read_budget=original_harness.budget,
        )
        original = queries._load_candidate_rows(
            **{
                **options,
                "membership_states": states,
                "read_budget": original_harness.budget,
            }
        )
    original_controls = counts.count("control")
    counts.clear()
    with connection.execute_wrapper(observe):
        assert candidate_rows(native, **options) == original
    native_controls = counts.count("control")
    assert counts.count("data") == 1
    assert (original_harness.budget.queries, original_harness.budget.source_rows) == (
        2,
        31 + len(original),
    )
    assert (native.budget.queries, native.budget.source_rows) == (1, 1 + len(original))
    assert native_controls < original_controls
    candidate, cp = (
        _candidate_queryset(**options).query.get_compiler(using="default").as_sql()
    )
    combined, params = statement(native, options)
    plans = []
    with bounded_seed_read(
        using="default", deadline=monotonic() + 4.5, alias=tier is Tier.ALIAS
    ):
        with connection.cursor() as cursor:
            for sql, values in ((candidate, cp), (combined, params)):
                cursor.execute("EXPLAIN (ANALYZE, FORMAT JSON) " + sql, values)
                plans.append(cursor.fetchone()[0][0]["Plan"])
    nested = first_limit(find(plans[1], "CTE candidates"))
    assert subtree(plans[0]) == subtree(nested)
    from apps.knowledge_graph.models import CollectionGraphMembershipState

    CollectionGraphMembershipState.objects.filter(
        collection_id=states[0]["collection_id"]
    ).update(registry_epoch=99)
    with connection.cursor() as cursor:
        cursor.execute("EXPLAIN (ANALYZE, FORMAT JSON) " + combined, params)
        invalid = cursor.fetchone()[0][0]["Plan"]
    assert first_limit(find(invalid, "CTE candidates"))["Actual Loops"] == 0
    print(
        {
            "tier": tier.value,
            "collections": 31,
            "old_rows": original_harness.budget.source_rows,
            "new_rows": native.budget.source_rows,
            "old_control": original_controls,
            "new_control": native_controls,
            "reservation": native.maxima,
            "candidate_tree_equal": True,
        }
    )


def test_product31_ontology_and16_chunk_proofs_are_two_real_records(scoped_aliases):
    from apps.documents.models import TextChunk
    from apps.knowledge_graph.retrieval.extended_seed_repository import _chunk_documents
    from apps.knowledge_graph.retrieval.native_chunk_owner_proof import (
        require_chunk_owners,
    )
    from apps.knowledge_graph.retrieval.native_query_ontology import (
        load_native_ontology,
    )
    from apps.knowledge_graph.retrieval.query_ontology import (
        _load_selected_artifact_rows,
    )

    fixture = many(scoped_aliases)
    harness = StatementHarness()
    harness.states = lambda group: fixture[1]
    # This SQL-only differential deliberately claims no closed-owner eligibility.
    rows = _load_selected_artifact_rows(
        selected_artifact_ids=fixture[0].selected_artifact_ids, using="default"
    )
    assert len(rows) == 31
    # The fixture has no active YAML definition: both routes rej
    # ect after its real query.
    outcome = load_native_ontology(harness, group=object())
    assert outcome.ontology is None and outcome.selected_artifact_count == 31
    assert (harness.budget.queries, harness.budget.source_rows) == (2, 1)
    chunk = TextChunk.objects.get()
    made = [chunk]
    for i in range(1, 16):
        made.append(
            TextChunk.objects.create(
                content=chunk.content[i:],
                doc_id=chunk.doc_id,
                chunk_number=i,
                start_position=i,
                end_position=chunk.end_position,
                embedding=[0.0] * 1024,
            )
        )
    requested = {c.pk: c.doc_id for c in made}
    harness.owner = SimpleNamespace(
        plan=SimpleNamespace(references={c.pk: c for c in made})
    )
    assert _chunk_documents(chunk_ids=tuple(requested), using="default") == requested
    require_chunk_owners(harness, requested=requested, using="default")
    assert (harness.budget.queries, harness.budget.source_rows) == (3, 2)
    assert harness.maxima == [1, 1]
    print(
        {
            "ontology_input": 31,
            "artifact_proof_rows": 1,
            "active_yaml_returned": 0,
            "chunks": 16,
            "chunk_proof_rows": 1,
            "queries": 3,
            "rows": 2,
        }
    )
