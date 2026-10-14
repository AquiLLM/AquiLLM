"""Database parity for the frozen HEAD 31886022 direct seed query."""

import os
from dataclasses import replace
from time import monotonic
from uuid import uuid4

import pytest

from apps.knowledge_graph.retrieval import direct_seed_queries
from apps.knowledge_graph.retrieval.direct_seed_contracts import DirectResolutionTier
from apps.knowledge_graph.tests import (
    _direct_seed_queries_head318_oracle as old_queries,
)

pytest_plugins = ["apps.knowledge_graph.tests.test_direct_seed_sql_postgres"]

pytestmark = [
    pytest.mark.django_db,
    pytest.mark.skipif(
        os.environ.get("KG_REQUIRE_POSTGRES_TESTS") != "1",
        reason="explicit PostgreSQL integration opt-in required",
    ),
]


def _options(scoped_aliases, tier=DirectResolutionTier.ALIAS, **changes):
    scope, states, _, _ = scoped_aliases
    lookup_fields = {
        DirectResolutionTier.ALIAS: (
            "document_links__document_entity__mention_links__mention__normalized_text"
        ),
        DirectResolutionTier.NAME: "normalized_label",
        DirectResolutionTier.IDENTIFIER: "identifier",
        DirectResolutionTier.EMBEDDING: "embedding",
    }
    options = dict(
        tier=tier,
        lookup="atlas" if tier is not DirectResolutionTier.IDENTIFIER else "atlas-id",
        lookup_field=lookup_fields[tier],
        embedding=None,
        model_signature="",
        minimum_similarity=0.5,
        ontology_type="model",
        membership_states=states,
        automatic_only=True,
        scope=scope,
        using="default",
        limit=4,
        deadline=monotonic() + 10,
    )
    options.update(changes)
    return options


def _same(options, expected):
    old = old_queries._load_candidate_rows(**options)
    new = direct_seed_queries._load_candidate_rows(**options)
    assert old == new
    assert [row.entity_id for row in new] == expected


@pytest.mark.parametrize(
    "tier",
    [
        DirectResolutionTier.NAME,
        DirectResolutionTier.ALIAS,
        DirectResolutionTier.IDENTIFIER,
        DirectResolutionTier.EMBEDDING,
    ],
)
def test_nonempty_tiers_keep_ordered_results_and_sentinel(scoped_aliases, tier):
    from django.db import connection

    _, _, entities, _ = scoped_aliases
    if tier is DirectResolutionTier.IDENTIFIER:
        with connection.cursor() as cursor:
            cursor.execute(
                "UPDATE apps_knowledge_graph_collectionentity "
                "SET identifier = 'atlas-id' WHERE id = %s",
                [entities[0].pk],
            )
    if tier is DirectResolutionTier.EMBEDDING:
        vector = [1.0] + [0.0] * 1023
        with connection.cursor() as cursor:
            cursor.execute(
                "UPDATE apps_knowledge_graph_collectionentity "
                "SET embedding = %s::vector, "
                "embedding_model_signature = 'test-model', embedding_input_hash = %s "
                "WHERE id IN (%s, %s, %s)",
                [
                    "[" + ",".join(map(str, vector)) + "]",
                    "a" * 64,
                    *[entity.pk for entity in entities],
                ],
            )
    options = _options(scoped_aliases, tier)
    if tier is DirectResolutionTier.EMBEDDING:
        options.update(embedding=[1.0] + [0.0] * 1023, model_signature="test-model")
    expected = (
        [entities[0].pk]
        if tier is DirectResolutionTier.IDENTIFIER
        else [entity.pk for entity in entities[:2]]
    )
    _same(options, expected)
    _same({**options, "limit": 2}, expected)
    if len(expected) > 1:
        for loader in (
            old_queries._load_candidate_rows,
            direct_seed_queries._load_candidate_rows,
        ):
            with pytest.raises(ValueError, match="hard cap"):
                loader(**{**options, "limit": 1})


@pytest.mark.parametrize("changed_field", ["outcome", "resolver_version"])
def test_bad_provenance_link_cannot_borrow_alias(scoped_aliases, changed_field):
    from django.db import connection

    from apps.knowledge_graph.models import CollectionEntityDocumentLink

    _, _, entities, _ = scoped_aliases
    link = CollectionEntityDocumentLink.objects.get(collection_entity=entities[0])
    if changed_field == "outcome":
        # Update the coupled pair to a valid candidate link.
        with connection.cursor() as cursor:
            cursor.execute(
                "UPDATE apps_knowledge_graph_collectionentitydocumentlink "
                "SET status = 'suppressed', outcome = 'candidate', "
                "candidate_rank = 1 WHERE id = %s",
                [link.pk],
            )
    else:
        with connection.cursor() as cursor:
            cursor.execute(
                "UPDATE apps_knowledge_graph_collectionentitydocumentlink "
                "SET resolver_version = 'wrong-resolver' WHERE id = %s",
                [link.pk],
            )
    _same(_options(scoped_aliases), [entities[1].pk])


@pytest.mark.parametrize(
    "field,value",
    [
        ("registry_epoch", 2),
        ("membership_checksum", "a" * 64),
        ("resolver_version", "different"),
        ("resolution_config_checksum", "a" * 64),
    ],
)
def test_changed_live_membership_rejects_candidate(scoped_aliases, field, value):
    from apps.knowledge_graph.models import CollectionGraphMembershipState

    CollectionGraphMembershipState.objects.update(**{field: value})
    _same(_options(scoped_aliases), [])


def test_selected_document_and_type_exclusion(scoped_aliases):
    scope, _, _, _ = scoped_aliases
    _same(
        _options(
            scoped_aliases, scope=replace(scope, selected_document_ids=(uuid4(),))
        ),
        [],
    )
    _same(_options(scoped_aliases, ontology_type="other"), [])
