from __future__ import annotations

import uuid

import pytest
from django.contrib.auth.models import User
from django.db import IntegrityError, connection, transaction
from django.test.utils import CaptureQueriesContext
from django.utils import timezone

from apps.collections.models import (
    Collection,
    CollectionPermission,
    CollectionSchemaDraft,
    CollectionSchemaGenerationRun,
    CollectionSchemaVersion,
)
from apps.knowledge_graph.models import OntologyVersion


def _definitions(name: str = "paper") -> dict:
    return {
        "relations": [],
        "entities": [
            {
                "key": name,
                "origin": "collection",
                "change_state": "added",
                "capabilities": {
                    "editable_fields": ["description", "aliases"],
                    "removable": True,
                    "renameable": True,
                },
                "values": {
                    "name": name,
                    "description": f"A {name}.",
                    "aliases": [],
                    "default_retrieval_weight": 1.0,
                    "default_suppression_policy": "never",
                    "default_suppression_threshold": 0.0,
                },
            }
        ],
    }


def _ontology(version: str, checksum: str) -> OntologyVersion:
    return OntologyVersion.objects.create(
        kind=OntologyVersion.Kind.GRAPH,
        version=version,
        checksum=checksum,
        metadata={},
    )


@pytest.mark.django_db
def test_collection_has_at_most_one_shared_schema_draft():
    user = User.objects.create_user(username="schema-draft-owner")
    collection = Collection.objects.create(name="Schema draft collection")
    CollectionSchemaDraft.objects.create(
        collection=collection,
        definitions=_definitions(),
        last_editor=user,
    )

    with pytest.raises(IntegrityError):
        with transaction.atomic():
            CollectionSchemaDraft.objects.create(
                collection=collection,
                definitions=_definitions("author"),
                last_editor=user,
            )


@pytest.mark.django_db
def test_published_schema_identity_is_unique_and_snapshot_is_immutable():
    user = User.objects.create_user(username="schema-publisher")
    collection = Collection.objects.create(name="Published schema collection")
    with pytest.raises(IntegrityError):
        with transaction.atomic():
            CollectionSchemaVersion.objects.create(
                collection=collection,
                version=1,
                checksum="0" * 64,
                definitions=_definitions(),
                published_by=user,
            )
    first = CollectionSchemaVersion.objects.create(
        collection=collection,
        version=1,
        checksum="a" * 64,
        definitions=_definitions(),
        ontology_version=_ontology("9.0.0+schema.model.1", "d" * 64),
        published_by=user,
        summary="Initial schema",
    )

    with pytest.raises(IntegrityError):
        with transaction.atomic():
            CollectionSchemaVersion.objects.create(
                collection=collection,
                version=1,
                checksum="b" * 64,
                definitions=_definitions("author"),
                ontology_version=_ontology("9.0.0+schema.model.2", "e" * 64),
                published_by=user,
            )
    with pytest.raises(IntegrityError):
        with transaction.atomic():
            CollectionSchemaVersion.objects.create(
                collection=collection,
                version=2,
                checksum="a" * 64,
                definitions=_definitions("author"),
                ontology_version=_ontology("9.0.0+schema.model.3", "f" * 64),
                published_by=user,
            )

    first.summary = "Rewritten history"
    with pytest.raises(ValueError, match="immutable"):
        first.save()
    with pytest.raises(ValueError, match="immutable"):
        CollectionSchemaVersion.objects.filter(pk=first.pk).update(
            summary="Rewritten through queryset"
        )


@pytest.mark.django_db
def test_published_schema_uses_mutable_collection_head_without_mutating_snapshots():
    user = User.objects.create_user(username="schema-head-publisher")
    collection = Collection.objects.create(name="Schema head collection")
    first = CollectionSchemaVersion.objects.create(
        collection=collection,
        version=1,
        checksum="1" * 64,
        definitions=_definitions(),
        ontology_version=_ontology("9.1.0+schema.head.1", "4" * 64),
        published_by=user,
    )
    second = CollectionSchemaVersion.objects.create(
        collection=collection,
        version=2,
        checksum="2" * 64,
        definitions=_definitions("author"),
        ontology_version=_ontology("9.1.0+schema.head.2", "5" * 64),
        published_by=user,
    )

    collection.current_schema_version = first
    collection.save(update_fields=("current_schema_version",))
    collection.current_schema_version = second
    collection.save(update_fields=("current_schema_version",))

    collection.refresh_from_db()
    assert collection.current_schema_version == second
    first.summary = "Mutable through the head"
    with pytest.raises(ValueError, match="immutable"):
        first.save()


@pytest.mark.django_db
def test_current_published_schema_rejects_direct_instance_delete():
    user = User.objects.create_user(username="schema-current-delete-publisher")
    collection = Collection.objects.create(name="Current schema delete collection")
    version = CollectionSchemaVersion.objects.create(
        collection=collection,
        version=1,
        checksum="8" * 64,
        definitions=_definitions(),
        ontology_version=_ontology("9.1.1+schema.current.delete", "9" * 64),
        published_by=user,
    )
    collection.current_schema_version = version
    collection.save(update_fields=("current_schema_version",))

    with pytest.raises(ValueError, match="immutable"):
        version.delete()

    collection.refresh_from_db()
    assert collection.current_schema_version == version
    assert CollectionSchemaVersion.objects.filter(pk=version.pk).exists()


@pytest.mark.django_db
def test_historical_published_schema_rejects_direct_queryset_delete():
    user = User.objects.create_user(username="schema-history-delete-publisher")
    collection = Collection.objects.create(name="Historical schema delete collection")
    historical = CollectionSchemaVersion.objects.create(
        collection=collection,
        version=1,
        checksum="a" * 64,
        definitions=_definitions(),
        ontology_version=_ontology("9.1.2+schema.history.delete.1", "b" * 64),
        published_by=user,
    )
    current = CollectionSchemaVersion.objects.create(
        collection=collection,
        version=2,
        checksum="c" * 64,
        definitions=_definitions("author"),
        ontology_version=_ontology("9.1.2+schema.history.delete.2", "d" * 64),
        published_by=user,
    )
    collection.current_schema_version = current
    collection.save(update_fields=("current_schema_version",))

    with pytest.raises(ValueError, match="immutable"):
        CollectionSchemaVersion.objects.filter(pk=historical.pk).delete()

    collection.refresh_from_db()
    assert collection.current_schema_version == current
    assert CollectionSchemaVersion.objects.filter(pk=historical.pk).exists()


@pytest.mark.django_db
def test_collection_delete_cascades_its_current_schema_without_head_cycle():
    user = User.objects.create_user(username="schema-delete-publisher")
    collection = Collection.objects.create(name="Schema delete collection")
    version = CollectionSchemaVersion.objects.create(
        collection=collection,
        version=1,
        checksum="6" * 64,
        definitions=_definitions(),
        ontology_version=_ontology("9.2.0+schema.delete.1", "7" * 64),
        published_by=user,
    )
    historical = CollectionSchemaVersion.objects.create(
        collection=collection,
        version=2,
        checksum="e" * 64,
        definitions=_definitions("author"),
        ontology_version=_ontology("9.2.0+schema.delete.2", "f" * 64),
        published_by=user,
    )
    collection.current_schema_version = version
    collection.save(update_fields=("current_schema_version",))

    collection.delete()

    assert not Collection.objects.filter(pk=collection.pk).exists()
    assert not CollectionSchemaVersion.objects.filter(pk=version.pk).exists()
    assert not CollectionSchemaVersion.objects.filter(pk=historical.pk).exists()
