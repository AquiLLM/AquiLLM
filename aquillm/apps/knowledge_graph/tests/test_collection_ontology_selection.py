from __future__ import annotations

import pytest
import yaml


def _ontology_yaml(version: str, entity: str, relation: str) -> str:
    return yaml.safe_dump(
        {
            "version": version,
            "entity_types": [
                {
                    "name": entity,
                    "description": f"A {entity}.",
                    "aliases": [],
                    "default_retrieval_weight": 1.0,
                    "default_suppression_policy": "never",
                    "default_suppression_threshold": 0.0,
                }
            ],
            "relations": [
                {
                    "name": relation,
                    "description": f"A {relation} relation.",
                    "direction": "directed",
                    "allowed_head_types": [entity],
                    "allowed_tail_types": [entity],
                }
            ],
        },
        sort_keys=True,
    )


@pytest.mark.django_db(transaction=True)
def test_collections_select_their_ontology_and_schema_less_collection_uses_global():
    from apps.collections.models import Collection
    from apps.knowledge_graph.models import OntologyVersion
    from apps.knowledge_graph.services.ontology import (
        activate_collection_ontology,
        activate_ontology,
        collection_ontology,
        load_ontology_yaml,
    )

    collection_a = Collection.objects.create(name="Ontology collection A")
    collection_b = Collection.objects.create(name="Ontology collection B")
    schema_less = Collection.objects.create(name="Schema-less collection")
    global_definition = load_ontology_yaml(
        _ontology_yaml("1.0.0", "global_entity", "global_relation")
    )
    definition_a = load_ontology_yaml(
        _ontology_yaml("1.0.0+collection.a.1", "entity_a", "relation_a")
    )
    definition_b = load_ontology_yaml(
        _ontology_yaml("1.0.0+collection.b.1", "entity_b", "relation_b")
    )

    global_record = activate_ontology(global_definition)
    record_a = activate_collection_ontology(collection_a.pk, definition_a)
    record_b = activate_collection_ontology(collection_b.pk, definition_b)

    assert set(collection_ontology(collection_a.pk).entity_types) == {"entity_a"}
    assert set(collection_ontology(collection_b.pk).entity_types) == {"entity_b"}
    assert set(collection_ontology(schema_less.pk).entity_types) == {"global_entity"}
    global_record.refresh_from_db()
    assert global_record.status == OntologyVersion.Status.ACTIVE
    assert record_a.metadata["collection_id"] == collection_a.pk
    assert record_b.metadata["collection_id"] == collection_b.pk


@pytest.mark.django_db(transaction=True)
def test_collection_activation_supersedes_only_same_collection():
    from apps.collections.models import Collection
    from apps.knowledge_graph.models import OntologyVersion
    from apps.knowledge_graph.services.ontology import (
        activate_collection_ontology,
        activate_ontology,
        load_ontology_yaml,
    )

    collection_a = Collection.objects.create(name="Ontology replacement A")
    collection_b = Collection.objects.create(name="Ontology replacement B")
    global_record = activate_ontology(
        load_ontology_yaml(_ontology_yaml("2.0.0", "global_two", "global_edge"))
    )
    old_a = activate_collection_ontology(
        collection_a.pk,
        load_ontology_yaml(
            _ontology_yaml("2.0.0+collection.a.1", "old_a", "old_edge_a")
        ),
    )
    active_b = activate_collection_ontology(
        collection_b.pk,
        load_ontology_yaml(
            _ontology_yaml("2.0.0+collection.b.1", "active_b", "edge_b")
        ),
    )
    new_a = activate_collection_ontology(
        collection_a.pk,
        load_ontology_yaml(
            _ontology_yaml("2.0.0+collection.a.2", "new_a", "new_edge_a")
        ),
    )

    old_a.refresh_from_db()
    active_b.refresh_from_db()
    global_record.refresh_from_db()
    assert old_a.status == OntologyVersion.Status.SUPERSEDED
    assert new_a.status == OntologyVersion.Status.ACTIVE
    assert active_b.status == OntologyVersion.Status.ACTIVE
    assert global_record.status == OntologyVersion.Status.ACTIVE


@pytest.mark.django_db(transaction=True)
def test_global_activation_rejects_collection_scoped_identity_collision():
    from apps.collections.models import Collection
    from apps.knowledge_graph.models import OntologyVersion
    from apps.knowledge_graph.services.ontology import (
        OntologyValidationError,
        activate_collection_ontology,
        activate_ontology,
        deployment_ontology,
        load_ontology_yaml,
    )

    collection = Collection.objects.create(name="Ontology collision collection")
    fallback = activate_ontology(
        load_ontology_yaml(_ontology_yaml("3.0.0", "fallback", "fallback_edge"))
    )
    collection_definition = load_ontology_yaml(
        _ontology_yaml(
            "3.0.0+collection.collision",
            "collection_entity",
            "collection_edge",
        )
    )
    collection_record = activate_collection_ontology(
        collection.pk, collection_definition
    )

    with pytest.raises(OntologyValidationError, match="another identity"):
        activate_ontology(collection_definition)

    fallback.refresh_from_db()
    collection_record.refresh_from_db()
    assert fallback.status == OntologyVersion.Status.ACTIVE
    assert collection_record.status == OntologyVersion.Status.ACTIVE
    assert set(deployment_ontology().entity_types) == {"fallback"}
