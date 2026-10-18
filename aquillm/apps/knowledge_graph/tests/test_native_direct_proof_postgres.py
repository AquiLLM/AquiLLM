"""Atomic native direct lookup preserves the original candidate oracle."""

import pytest

from apps.knowledge_graph.retrieval import direct_seed_queries as queries
from apps.knowledge_graph.retrieval.direct_seed_contracts import (
    DirectResolutionTier as Tier,
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
def test_native_atomic_candidate_dispatch_has_mandatory_header(scoped_aliases, tier):
    from apps.knowledge_graph.retrieval import native_direct_seed_statement as native

    options = prepare_tier(scoped_aliases, tier)
    oracle = queries._load_candidate_rows(**options)
    from apps.knowledge_graph.tests import _direct_seed_queries_head318_oracle as frozen

    assert oracle == frozen._load_candidate_rows(**options)
    harness = StatementHarness()
    ledger = harness.budget
    result = native.candidate_rows(harness, **options)
    assert result == oracle
    assert (ledger.queries, ledger.source_rows) == (1, len(oracle) + 1)
    assert harness.maxima == [options["limit"] + 2]
    assert (
        native.candidate_rows(harness, **{**options, "ontology_type": "absent"}) == ()
    )
    assert (ledger.queries, ledger.source_rows) == (2, len(oracle) + 2)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "null",
        "foreign",
        "registry_epoch",
        "membership_checksum",
        "resolver_version",
        "resolution_config_checksum",
        "swapped",
    ],
)
def test_native_fresh_membership_proof_rejects_state_or_pair_drift(
    scoped_aliases, mutation
):
    from apps.knowledge_graph.models import CollectionGraphMembershipState as State
    from apps.knowledge_graph.retrieval.native_direct_seed_statement import (
        candidate_rows,
    )
    from apps.knowledge_graph.tests.test_direct_seed_exists_postgres import (
        _two_selected_states,
    )

    fixture = _two_selected_states(scoped_aliases)
    options = prepare_tier(fixture, Tier.NAME)
    first, second = fixture[1]
    if mutation == "missing":
        State.objects.filter(collection_id=first["collection_id"]).delete()
    elif mutation == "swapped":
        for target, source in ((first, second), (second, first)):
            State.objects.filter(collection_id=target["collection_id"]).update(
                **{f: source[f] for f in first if f != "collection_id"}
            )
    else:
        field = "active_artifact_id" if mutation in ("null", "foreign") else mutation
        value = (
            None
            if mutation == "null"
            else second["active_artifact_id"]
            if mutation == "foreign"
            else 99
            if mutation == "registry_epoch"
            else "changed"
        )
        State.objects.filter(collection_id=first["collection_id"]).update(
            **{field: value}
        )
    harness = StatementHarness()
    with pytest.raises(ValueError, match="membership proof"):
        candidate_rows(harness, **options)
    assert (harness.budget.queries, harness.budget.source_rows) == (1, 1)
    assert harness.failed


def test_mandatory_header_shape_order_and_types_reject_after_actual_debit(
    scoped_aliases,
):
    from apps.knowledge_graph.retrieval.native_direct_seed_statement import (
        statement,
        validate,
    )

    options = prepare_tier(scoped_aliases, Tier.NAME)
    harness = StatementHarness()
    sql, params = statement(harness, options)
    rows = harness.read(sql, params, using="default", maximum=6)
    assert len(rows) > 2
    for bad in (
        (),
        rows + rows[:1],
        (rows[1], *rows[:1]),
        (rows[0], *reversed(rows[1:])),
        ((False, *rows[0][1:]), *rows[1:]),
        ((0, True, *rows[0][2:]), *rows[1:]),
    ):
        with pytest.raises(ValueError):
            validate(bad, count=1, limit=4)
    assert harness.budget.source_rows == len(rows)


@pytest.mark.parametrize(
    "expected,current", [("a", "A"), ("\u00e9", "e\u0301"), ("a", "a ")]
)
def test_native_text_identity_matches_exact_python_equality(
    scoped_aliases, expected, current
):
    from apps.knowledge_graph.models import CollectionGraphMembershipState as State
    from apps.knowledge_graph.retrieval.native_direct_seed_statement import (
        candidate_rows,
    )

    options = prepare_tier(scoped_aliases, Tier.NAME)
    state = options["membership_states"][0]
    State.objects.filter(collection_id=state["collection_id"]).update(
        resolver_version=current
    )
    options["membership_states"] = ({**state, "resolver_version": expected},)
    harness = StatementHarness()
    with pytest.raises(ValueError, match="membership proof"):
        candidate_rows(harness, **options)
    assert expected != current and harness.budget.source_rows == 1


@pytest.mark.parametrize(
    "change",
    [
        "superseded",
        "alias_owner",
        "automatic",
        "embedding_signature",
        "embedding_threshold",
    ],
)
def test_original_witness_automatic_and_embedding_predicates(scoped_aliases, change):
    from apps.knowledge_graph.models import DocumentEntityMention, GraphArtifact
    from apps.knowledge_graph.retrieval.native_direct_seed_statement import (
        candidate_rows,
    )
    from apps.knowledge_graph.tests.test_direct_seed_exists_authority_postgres import (
        _canonical,
    )
    from apps.knowledge_graph.tests.test_direct_seed_exists_witness_postgres import _set

    tier = Tier.EMBEDDING if change.startswith("embedding") else Tier.ALIAS
    options = prepare_tier(scoped_aliases, tier)
    if change == "superseded":
        _set(
            GraphArtifact,
            options["scope"].selected_document_artifact_ids[0],
            status="superseded",
        )
    elif change == "alias_owner":
        _set(DocumentEntityMention, scoped_aliases[3][0].pk, resolver_version="changed")
    elif change == "automatic":
        _canonical(scoped_aliases[2][0], identity="1")
        _canonical(scoped_aliases[2][0], identity="2", resolver="different")
    elif change == "embedding_signature":
        options["model_signature"] = "different"
    else:
        options.update(embedding=[0.6, 0.8] + [0.0] * 1022, minimum_similarity=0.7)
    harness = StatementHarness()
    if change == "automatic":
        with pytest.raises(ValueError, match="automatic membership"):
            queries._load_candidate_rows(**options)
        with pytest.raises(ValueError, match="automatic membership"):
            candidate_rows(harness, **options)
    else:
        assert candidate_rows(harness, **options) == queries._load_candidate_rows(
            **options
        )


@pytest.mark.parametrize("size", [128, 129])
def test_native_header_and_overflow_are_charged_at_original_cap(scoped_aliases, size):
    from apps.knowledge_graph.retrieval.native_direct_seed_statement import (
        candidate_rows,
    )
    from apps.knowledge_graph.tests.native_metadata_test_support import (
        candidate_cardinality_fixture,
    )

    fixture = candidate_cardinality_fixture(scoped_aliases, size)
    options = {**prepare_tier(fixture, Tier.NAME), "limit": 128}
    harness = StatementHarness()
    if size == 128:
        assert len(candidate_rows(harness, **options)) == 128
    else:
        with pytest.raises(ValueError, match="hard cap"):
            candidate_rows(harness, **options)
    assert harness.maxima == [130]
    assert (harness.budget.queries, harness.budget.source_rows) == (1, size + 1)
    assert harness.budget._reserved_rows == 0 and not harness.budget._uncertain
