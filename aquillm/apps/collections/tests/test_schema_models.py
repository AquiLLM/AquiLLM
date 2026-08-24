from __future__ import annotations

import uuid

import pytest
from django.contrib.auth.models import User
from django.db import IntegrityError, transaction

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
def test_collection_has_at_most_one_queued_or_running_generation():
    user = User.objects.create_user(username="schema-generator")
    collection = Collection.objects.create(name="Generated schema collection")
    CollectionSchemaGenerationRun.objects.create(
        id=uuid.uuid4(),
        collection=collection,
        requested_by=user,
        status=CollectionSchemaGenerationRun.Status.QUEUED,
        source_signature="a" * 64,
    )

    with pytest.raises(IntegrityError):
        with transaction.atomic():
            CollectionSchemaGenerationRun.objects.create(
                id=uuid.uuid4(),
                collection=collection,
                requested_by=user,
                status=CollectionSchemaGenerationRun.Status.RUNNING,
                source_signature="b" * 64,
            )

    CollectionSchemaGenerationRun.objects.filter(collection=collection).update(
        status=CollectionSchemaGenerationRun.Status.SUCCEEDED
    )
    CollectionSchemaGenerationRun.objects.create(
        id=uuid.uuid4(),
        collection=collection,
        requested_by=user,
        status=CollectionSchemaGenerationRun.Status.QUEUED,
        source_signature="b" * 64,
    )


@pytest.mark.django_db
def test_new_collection_workspace_has_empty_published_schema_not_fixture_data():
    from apps.collections.services.schema import workspace_envelope

    user = User.objects.create_user(username="empty-schema-viewer")
    collection = Collection.objects.create(name="Fresh collection")
    CollectionPermission.objects.create(
        user=user,
        collection=collection,
        permission="VIEW",
    )

    workspace = workspace_envelope(collection, user)

    assert workspace["published"] == {
        "version": 0,
        "checksum": "",
        "entities": [],
        "relations": [],
    }
    assert workspace["draft"] is None


@pytest.mark.django_db
def test_generation_run_allows_system_requester():
    collection = Collection.objects.create(name="System generation collection")

    run = CollectionSchemaGenerationRun.objects.create(
        collection=collection,
        requested_by=None,
        status=CollectionSchemaGenerationRun.Status.QUEUED,
        source_signature="c" * 64,
    )

    assert run.requested_by is None


@pytest.mark.django_db
def test_write_generated_draft_requires_running_exact_base_and_completes_run():
    from apps.collections.services.schema import (
        SchemaGenerationDraftConflict,
        write_generated_draft,
    )

    user = User.objects.create_user(username="generation-writer")
    collection = Collection.objects.create(name="Generation write collection")
    draft = CollectionSchemaDraft.objects.create(
        collection=collection,
        revision=3,
        definitions=_definitions("paper"),
        last_editor=user,
    )
    queued = CollectionSchemaGenerationRun.objects.create(
        collection=collection,
        requested_by=user,
        status=CollectionSchemaGenerationRun.Status.QUEUED,
        source_signature="d" * 64,
        base_draft_revision=3,
    )
    with pytest.raises(ValueError, match="running"):
        write_generated_draft(queued.pk, _definitions("author"), {"entities": 1})

    queued.status = CollectionSchemaGenerationRun.Status.RUNNING
    queued.base_draft_revision = 2
    queued.save(update_fields=("status", "base_draft_revision", "updated_at"))
    with pytest.raises(SchemaGenerationDraftConflict):
        write_generated_draft(queued.pk, _definitions("author"), {"entities": 1})

    queued.base_draft_revision = 3
    queued.save(update_fields=("base_draft_revision", "updated_at"))
    written = write_generated_draft(
        queued.pk,
        {"entities": _definitions("author")["entities"], "relations": []},
        {"entities": 1},
    )

    written.refresh_from_db()
    queued.refresh_from_db()
    assert written.pk == draft.pk
    assert written.revision == 4
    assert written.definitions["entities"][0]["key"] == "author"
    assert queued.status == CollectionSchemaGenerationRun.Status.SUCCEEDED
    assert queued.statistics == {"entities": 1}
    assert queued.completed_at is not None
