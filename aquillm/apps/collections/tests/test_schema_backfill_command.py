from io import StringIO

import pytest
from django.contrib.auth.models import User
from django.core.management import call_command, get_commands

from apps.collections.models import (
    Collection,
    CollectionPermission,
    CollectionSchemaDraft,
    CollectionSchemaGenerationRun,
)


def test_missing_schema_backfill_command_is_registered():
    assert "generate_missing_collection_schemas" in get_commands()


@pytest.mark.django_db(transaction=True)
def test_backfill_queues_unchanged_empty_draft_and_skips_nonempty_draft(monkeypatch):
    editor = User.objects.create_user(username="schema-backfill-editor")
    empty = Collection.objects.create(name="Empty draft collection")
    nonempty = Collection.objects.create(name="Edited draft collection")
    for collection in (empty, nonempty):
        CollectionPermission.objects.create(
            collection=collection,
            user=editor,
            permission="EDIT",
        )
    empty_draft = CollectionSchemaDraft.objects.create(
        collection=empty,
        definitions={"entities": [], "relations": []},
        last_editor=editor,
    )
    CollectionSchemaDraft.objects.create(
        collection=nonempty,
        definitions={"entities": [{"key": "paper"}], "relations": []},
        last_editor=editor,
    )
    monkeypatch.setattr(
        "apps.collections.management.commands.generate_missing_collection_schemas._locked_collection_source_signature",
        lambda collection_id: f"{collection_id:064x}",
        raising=False,
    )
    enqueued = []
    monkeypatch.setattr(
        "apps.collections.management.commands.generate_missing_collection_schemas.enqueue_schema_generation",
        enqueued.append,
        raising=False,
    )
    output = StringIO()

    call_command(
        "generate_missing_collection_schemas",
        "--all",
        "--yes",
        stdout=output,
    )

    run = CollectionSchemaGenerationRun.objects.get()
    assert run.collection_id == empty.pk
    assert run.requested_by_id == editor.pk
    assert run.base_draft_id == empty_draft.pk
    assert run.base_draft_revision == empty_draft.revision
    assert enqueued == [str(run.pk)]
    assert output.getvalue().strip() == "queued=1 reused=0 skipped=1"


@pytest.mark.django_db(transaction=True)
def test_backfill_uses_inherited_editor_for_collection_without_a_draft(monkeypatch):
    manager = User.objects.create_user(username="schema-backfill-manager")
    parent = Collection.objects.create(name="Schema parent")
    child = Collection.objects.create(name="Schema child", parent=parent)
    CollectionPermission.objects.create(
        collection=parent,
        user=manager,
        permission="MANAGE",
    )
    monkeypatch.setattr(
        "apps.collections.management.commands.generate_missing_collection_schemas._locked_collection_source_signature",
        lambda collection_id: f"{collection_id:064x}",
    )
    enqueued = []
    monkeypatch.setattr(
        "apps.collections.management.commands.generate_missing_collection_schemas.enqueue_schema_generation",
        enqueued.append,
    )

    call_command("generate_missing_collection_schemas", "--collection", str(child.pk))

    run = CollectionSchemaGenerationRun.objects.get(collection=child)
    assert run.requested_by_id == manager.pk
    assert run.base_draft_id is None
    assert run.base_draft_revision is None
    assert enqueued == [str(run.pk)]


@pytest.mark.django_db(transaction=True)
def test_backfill_rebinds_legacy_active_run_to_empty_draft(monkeypatch):
    editor = User.objects.create_user(username="schema-backfill-legacy-editor")
    collection = Collection.objects.create(name="Legacy generation collection")
    CollectionPermission.objects.create(
        collection=collection,
        user=editor,
        permission="EDIT",
    )
    draft = CollectionSchemaDraft.objects.create(
        collection=collection,
        definitions={"entities": [], "relations": []},
        last_editor=editor,
    )
    run = CollectionSchemaGenerationRun.objects.create(
        collection=collection,
        requested_by=editor,
        source_signature="a" * 64,
        base_draft_id=None,
        base_draft_revision=None,
    )
    monkeypatch.setattr(
        "apps.collections.management.commands.generate_missing_collection_schemas._locked_collection_source_signature",
        lambda collection_id: "a" * 64,
    )
    enqueued = []
    monkeypatch.setattr(
        "apps.collections.management.commands.generate_missing_collection_schemas.enqueue_schema_generation",
        enqueued.append,
    )
    output = StringIO()

    call_command(
        "generate_missing_collection_schemas",
        "--collection",
        str(collection.pk),
        stdout=output,
    )

    run.refresh_from_db()
    assert run.base_draft_id == draft.pk
    assert run.base_draft_revision == draft.revision
    assert enqueued == [str(run.pk)]
    assert output.getvalue().strip() == "queued=0 reused=1 skipped=0"
