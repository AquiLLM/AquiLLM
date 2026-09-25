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
from apps.collections.tests.test_schema_models import (
    _definitions,
    _ontology,
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
    lease_token = uuid.uuid4()
    lease_expires_at = timezone.now()

    run = CollectionSchemaGenerationRun.objects.create(
        collection=collection,
        requested_by=None,
        status=CollectionSchemaGenerationRun.Status.QUEUED,
        source_signature="c" * 64,
        base_draft_id=uuid.uuid4(),
        lease_token=lease_token,
        lease_expires_at=lease_expires_at,
    )

    assert run.requested_by is None
    assert run.base_draft_id is not None
    assert run.lease_token == lease_token
    assert run.lease_expires_at == lease_expires_at


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
        base_draft_id=draft.pk,
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


@pytest.mark.django_db
def test_generated_draft_rejects_same_revision_replacement_identity():
    from apps.collections.services.schema import (
        SchemaGenerationDraftConflict,
        write_generated_draft,
    )

    user = User.objects.create_user(username="generation-fenced-writer")
    collection = Collection.objects.create(name="Generation fenced collection")
    original = CollectionSchemaDraft.objects.create(
        collection=collection,
        revision=3,
        definitions=_definitions("paper"),
        last_editor=user,
    )
    run = CollectionSchemaGenerationRun.objects.create(
        collection=collection,
        requested_by=user,
        status=CollectionSchemaGenerationRun.Status.RUNNING,
        source_signature="e" * 64,
        base_draft_id=original.pk,
        base_draft_revision=3,
    )
    original.delete()
    replacement = CollectionSchemaDraft.objects.create(
        collection=collection,
        revision=3,
        definitions=_definitions("replacement"),
        last_editor=user,
    )

    with pytest.raises(SchemaGenerationDraftConflict):
        write_generated_draft(run.pk, _definitions("generated"), {"entities": 1})

    replacement.refresh_from_db()
    run.refresh_from_db()
    assert replacement.revision == 3
    assert replacement.definitions["entities"][0]["key"] == "replacement"
    assert run.status == CollectionSchemaGenerationRun.Status.RUNNING


@pytest.mark.django_db
def test_generated_draft_locks_collection_before_generation_run():
    from apps.collections.services.schema import write_generated_draft

    user = User.objects.create_user(username="generation-lock-order-writer")
    collection = Collection.objects.create(name="Generation lock order collection")
    draft = CollectionSchemaDraft.objects.create(
        collection=collection,
        revision=2,
        definitions=_definitions("paper"),
        last_editor=user,
    )
    run = CollectionSchemaGenerationRun.objects.create(
        collection=collection,
        requested_by=user,
        status=CollectionSchemaGenerationRun.Status.RUNNING,
        source_signature="f" * 64,
        base_draft_id=draft.pk,
        base_draft_revision=2,
    )

    with CaptureQueriesContext(connection) as queries:
        write_generated_draft(run.pk, _definitions("generated"), {"entities": 1})

    locking_queries = [
        query["sql"].lower()
        for query in queries
        if "for update" in query["sql"].lower()
    ]
    collection_lock = next(
        index
        for index, sql in enumerate(locking_queries)
        if 'from "aquillm_collection"' in sql
    )
    run_lock = next(
        index
        for index, sql in enumerate(locking_queries)
        if 'from "apps_collections_collectionschemagenerationrun"' in sql
    )
    assert collection_lock < run_lock
