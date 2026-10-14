"""Status and canonical authority parity for the EXISTS candidate."""

import os

import pytest

from apps.knowledge_graph.tests.test_direct_seed_exists_postgres import _options, _same
from apps.knowledge_graph.tests.test_direct_seed_exists_witness_postgres import _set

pytest_plugins = ["apps.knowledge_graph.tests.test_direct_seed_sql_postgres"]

pytestmark = [
    pytest.mark.django_db,
    pytest.mark.skipif(
        os.environ.get("KG_REQUIRE_POSTGRES_TESTS") != "1",
        reason="explicit PostgreSQL integration opt-in required",
    ),
]


@pytest.mark.parametrize(
    "owner,field,value,expected_two",
    [
        ("entity", "status", "superseded", False),
        ("entity", "entity_type", "other", False),
        ("document", "status", "superseded", False),
        ("mention", "entity_type", "other", False),
        ("assignment", "status", "superseded", False),
        ("source", "status", "building", True),
        ("source", "evaluation_only", True, True),
        ("source", "ontology_checksum", "a" * 64, True),
        ("collection_artifact", "ontology_checksum", "a" * 64, True),
    ],
)
def test_selected_and_excluded_status_type_ontology(
    scoped_aliases, owner, field, value, expected_two
):
    from apps.knowledge_graph.models import (
        CollectionEntity,
        DocumentEntity,
        DocumentEntityMention,
        EntityMention,
        GraphArtifact,
        GraphRebuildRequest,
    )

    scope, _, entities, assignments = scoped_aliases
    owner_map = {
        "entity": (CollectionEntity, entities[0].pk),
        "document": (DocumentEntity, assignments[0].document_entity_id),
        "mention": (EntityMention, assignments[0].mention_id),
        "assignment": (DocumentEntityMention, assignments[0].pk),
        "source": (GraphArtifact, scope.selected_document_artifact_ids[0]),
        "collection_artifact": (GraphArtifact, scope.selected_artifact_ids[0]),
    }
    model, pk = owner_map[owner]
    if owner == "source" and field == "evaluation_only":
        # Evaluation artifacts must be noncurrent under kg_artifact_eval_noncurrent.
        # Superseded source artifacts would otherwise remain query-eligible.
        request = GraphRebuildRequest.objects.create(
            scope_type="collection",
            scope_id=str(scope.selected_collection_ids[0]),
            collection_count=1,
            evaluation_only=True,
        )
        _set(
            model,
            pk,
            status="superseded",
            evaluation_only=True,
            rebuild_request_id=request.pk,
        )
    else:
        _set(model, pk, **{field: value})
    expected = [] if expected_two else [entities[1].pk]
    _same(_options(scoped_aliases), expected)


def _canonical(
    entity, *, identity, resolver="canonical-v1", entity_type="model", signature=""
):
    from apps.knowledge_graph.models import CanonicalEntity, CanonicalEntityLink

    target = CanonicalEntity.objects.create(
        identity_key=identity * 64,
        resolver_version=resolver,
        label="Atlas",
        normalized_label="atlas",
        entity_type=entity_type,
        version_signature=signature,
    )
    link = CanonicalEntityLink.objects.create(
        collection_entity=entity,
        canonical_entity=target,
        score=1.0,
        method="exact_name_or_alias",
        resolver_version=resolver,
        outcome="automatic",
        status="active",
        reason="fixture automatic",
    )
    return target, link


@pytest.mark.parametrize(
    "field,value",
    [
        ("version_signature", "incompatible"),
        ("entity_type", "other"),
        ("resolver_version", "other"),
    ],
)
def test_incompatible_canonical_link_remains_unmapped(scoped_aliases, field, value):
    from apps.knowledge_graph.models import CanonicalEntity
    from apps.knowledge_graph.retrieval import direct_seed_queries

    _, _, entities, _ = scoped_aliases
    target, _ = _canonical(entities[0], identity="1")
    _same(_options(scoped_aliases), [entities[0].pk, entities[1].pk])
    before = direct_seed_queries._load_candidate_rows(**_options(scoped_aliases))
    assert before[0].automatic_canonical_entity_id == target.pk
    _set(CanonicalEntity, target.pk, **{field: value})
    _same(_options(scoped_aliases), [entities[0].pk, entities[1].pk])
    assert all(
        row.automatic_canonical_entity_id is None
        for row in (
            direct_seed_queries._load_candidate_rows(**_options(scoped_aliases))
        )
    )


def test_active_automatic_count_includes_incompatible_link(scoped_aliases):
    from django.db.models import F

    from apps.knowledge_graph.models import CanonicalEntity, CanonicalEntityLink
    from apps.knowledge_graph.retrieval import direct_seed_queries
    from apps.knowledge_graph.tests import (
        _direct_seed_queries_head318_oracle as old_queries,
    )

    _, _, entities, _ = scoped_aliases
    first_target, first_link = _canonical(entities[0], identity="2")
    second_target, second_link = _canonical(
        entities[0], identity="3", resolver="different-resolver"
    )
    _set(CanonicalEntity, second_target.pk, version_signature="incompatible")
    second_target.refresh_from_db()
    assert first_target.resolver_version == first_link.resolver_version
    assert second_target.resolver_version == second_link.resolver_version
    assert first_link.resolver_version != second_link.resolver_version
    assert second_target.version_signature != entities[0].version_signature
    active = CanonicalEntityLink.objects.filter(
        collection_entity=entities[0],
        status="active",
        outcome="automatic",
        canonical_entity__status="active",
    )
    assert active.count() == 2
    assert (
        active.filter(
            canonical_entity__resolver_version=F("resolver_version"),
            canonical_entity__entity_type=entities[0].entity_type,
            canonical_entity__version_signature=entities[0].version_signature,
        ).count()
        == 1
    )
    for loader in (
        old_queries._load_candidate_rows,
        direct_seed_queries._load_candidate_rows,
    ):
        with pytest.raises(ValueError, match="automatic membership is not unique"):
            loader(**_options(scoped_aliases))
