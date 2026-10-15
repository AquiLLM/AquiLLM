"""Native statement fresh-call accounting, cancellation and settings restoration."""

import pytest
from django.db import connection

from apps.knowledge_graph.models import CollectionGraphMembershipState as State
from apps.knowledge_graph.retrieval.direct_seed_contracts import (
    DirectResolutionTier as Tier,
)
from apps.knowledge_graph.retrieval.direct_seed_sql import DirectSeedReadTimeout
from apps.knowledge_graph.retrieval.native_direct_seed_statement import candidate_rows
from apps.knowledge_graph.tests.native_metadata_test_support import (
    StatementHarness,
    budget,
    prepare_tier,
)
from apps.knowledge_graph.tests.test_direct_seed_sql_postgres import _settings
from apps.knowledge_graph.tests.test_direct_seed_sql_postgres import (
    scoped_aliases as scoped_aliases,
)

pytestmark = pytest.mark.django_db


def test_completed_first_tier_never_authorizes_fresh_next_tier(scoped_aliases):
    options = prepare_tier(scoped_aliases, Tier.NAME)
    harness = StatementHarness()
    assert candidate_rows(harness, **options)
    State.objects.filter(
        collection_id=options["membership_states"][0]["collection_id"]
    ).update(registry_epoch=99)
    with pytest.raises(ValueError, match="membership proof"):
        candidate_rows(
            harness,
            **{
                **options,
                "tier": Tier.ALIAS,
                "lookup_field": (
                    "document_links__document_entity__"
                    "mention_links__mention__normalized_text"
                ),
            },
        )
    before = (harness.budget.queries, harness.budget.source_rows)
    with pytest.raises(ValueError):
        candidate_rows(harness, **options)
    assert before == (harness.budget.queries, harness.budget.source_rows)


@pytest.mark.parametrize("change", ["timeout", "unknown", "missing_payload"])
def test_real_native_cursor_uncertainty_poison_and_restore(scoped_aliases, change):
    options = prepare_tier(scoped_aliases, Tier.ALIAS)
    harness = StatementHarness(budget(0.06 if change == "timeout" else 4.5))
    before = _settings()
    dispatched = []
    if change == "missing_payload":

        def missing(*args, **kwargs):
            # The unchanged meter itself sees the mismatching returned payload.
            from apps.knowledge_graph.retrieval.graph_seed_reads_v2 import (
                _metered_seed_rows,
            )

            def read():
                with connection.cursor() as cursor:
                    cursor.execute(args[0], args[1])
                    cursor.fetchall()
                return ()

            return _metered_seed_rows(
                read,
                using=kwargs["using"],
                maximum=kwargs["maximum"],
                budget=harness.budget,
            )

        harness.read = missing

    def observe(execute, sql, params, many, context):
        native = sql.startswith("WITH expected(")
        if native:
            dispatched.append(True)
            assert harness.budget._reserved_rows == options["limit"] + 2
        if native and change == "timeout":
            return execute("SELECT pg_sleep(2)", (), many, context)
        if native and change == "unknown":

            class Unknown:
                rowcount = -1

            context["cursor"] = Unknown()
        return execute(sql, params, many, context)

    expected = DirectSeedReadTimeout if change == "timeout" else ValueError
    with connection.execute_wrapper(observe), pytest.raises(expected):
        candidate_rows(harness, **options)
    assert dispatched == [True] and harness.budget._uncertain and harness.failed
    assert _settings() == before
    debits = (
        harness.budget.queries,
        harness.budget.source_rows,
        harness.budget._reserved_rows,
    )
    with pytest.raises((ValueError, TimeoutError)):
        candidate_rows(harness, **options)
    assert debits == (
        harness.budget.queries,
        harness.budget.source_rows,
        harness.budget._reserved_rows,
    )
