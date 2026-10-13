"""Staged PostgreSQL upgrades retain legacy bytes and refuse lossy reversal."""
from importlib import import_module
from uuid import uuid4

import pytest
from django.apps import apps
from django.db import connection, transaction
from django.db.migrations.executor import MigrationExecutor

from apps.documents.models import TextChunk, DocumentRepresentationState
from .representation_support import retained as retained, upload as upload, activate
from .test_representation_schema import capture

pytestmark = pytest.mark.django_db(transaction=True)
BEFORE = [("apps_documents", "0012_chunk_search_projection_gin_v1")]
ADDITIVE = [("apps_documents", "0013_representation_generations_v1")]


def test_populated_legacy_roundtrip_preserves_good_bad_and_orphan_rows(retained):
    doc, output, old = retained
    leaves = MigrationExecutor(connection).loader.graph.leaf_nodes()
    try:
        executor = MigrationExecutor(connection)
        executor.migrate(BEFORE)
        model = executor.loader.project_state(BEFORE).apps.get_model("apps_documents", "TextChunk")
        bad = model.objects.create(doc_id=doc.id, content="saved invalid slice", chunk_number=1,
            start_position=99, end_position=118)
        orphan = model.objects.create(doc_id=uuid4(), content="saved orphan", chunk_number=0,
            start_position=0, end_position=12)
        MigrationExecutor(connection).migrate(leaves)
        assert not DocumentRepresentationState.objects.exists()
        for pk, content in ((old.pk, old.content), (bad.pk, bad.content), (orphan.pk, orphan.content)):
            row = TextChunk.objects.get(pk=pk)
            assert row.content == content and row.representation_generation_id is None
            assert row.legacy_source_snapshot_id is None and row.legacy_context_status == "unbound"
        # The additive migration invents neither history nor artifact associations.
        assert activate(doc, output) == "activated"
        old.refresh_from_db()
        bad = TextChunk.objects.get(pk=bad.pk)
        assert old.legacy_context_status == "verified"
        assert bad.legacy_context_status == "unavailable" and bad.content == "saved invalid slice"
        assert TextChunk.objects.get(pk=orphan.pk).legacy_source_snapshot_id is None
    finally:
        MigrationExecutor(connection).migrate(leaves)


def test_unused_schema_reversal_and_leaf_restoration():
    leaves = MigrationExecutor(connection).loader.graph.leaf_nodes()
    try:
        MigrationExecutor(connection).migrate(BEFORE)
        with connection.cursor() as cursor:
            cursor.execute("SELECT to_regclass('aquillm_representationgeneration')")
            assert cursor.fetchone() == (None,)
        MigrationExecutor(connection).migrate(leaves)
        assert not DocumentRepresentationState.objects.exists()
    finally:
        MigrationExecutor(connection).migrate(leaves)


def test_used_schema_reverse_refuses_before_erasing_exact_history(retained):
    doc, _, old = retained
    state, snapshot = capture(doc)
    leaves = MigrationExecutor(connection).loader.graph.leaf_nodes()
    try:
        with pytest.raises(RuntimeError, match="representation_history_in_use"):
            MigrationExecutor(connection).migrate(BEFORE)
        assert DocumentRepresentationState.objects.get(pk=state.pk).document_id == doc.id
        assert TextChunk.objects.get(pk=old.pk).legacy_source_snapshot_id == snapshot.pk
    finally:
        MigrationExecutor(connection).migrate(leaves)


def test_state_only_reverse_refuses_before_removing_installed_guards(retained):
    from apps.documents.services.representation_history import locked_state
    from apps.knowledge_graph.graph.invalidation import DocumentLifecycleRef, locked_document_lifecycle_row

    doc, _, _ = retained
    with locked_document_lifecycle_row(DocumentLifecycleRef(doc._meta.label_lower, doc.pkid, doc.id),
                                      (doc.collection_id,), using="default") as (current, _):
        state = locked_state(current, using="default")
    leaves = MigrationExecutor(connection).loader.graph.leaf_nodes()
    try:
        with pytest.raises(RuntimeError, match="representation_history_in_use"):
            MigrationExecutor(connection).migrate(BEFORE)
        applied = ("apps_documents", "0014_generation_chunk_constraints_v1") in (
            MigrationExecutor(connection).loader.applied_migrations)
        with connection.cursor() as cursor:
            cursor.execute("""SELECT to_regprocedure('representation_state_v1()') IS NOT NULL,
                EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'representation_state_v1'
                    AND tgrelid = 'aquillm_representationstate'::regclass)""")
            function_present, trigger_present = cursor.fetchone()
        assert (applied, function_present, trigger_present) == (True, True, True)
        assert DocumentRepresentationState.objects.get(pk=state.pk).document_id == doc.id
    finally:
        MigrationExecutor(connection).migrate(leaves)


def test_staged_validator_rejects_actual_duplicate_legacy_rows(retained):
    doc, _, old = retained
    leaves = MigrationExecutor(connection).loader.graph.leaf_nodes()
    try:
        MigrationExecutor(connection).migrate(ADDITIVE)
        # Model a damaged predecessor schema only inside a rolled-back transaction.
        # No trigger is disabled and the original constraints return on rollback.
        with pytest.raises(RuntimeError, match="representation_chunk_duplicates"), transaction.atomic():
            with connection.cursor() as cursor:
                for name in ("uniqe_chunk_per_document", "unique_chunk_position_per_document"):
                    cursor.execute(f'ALTER TABLE aquillm_textchunk DROP CONSTRAINT "{name}"')
            TextChunk.objects.create(doc_id=doc.id, content=old.content, chunk_number=0,
                start_position=old.start_position, end_position=old.end_position)
            migration = import_module("apps.documents.migrations.0014_generation_chunk_constraints_v1")
            with connection.schema_editor() as editor:
                migration.validate_rows(apps, editor)
    finally:
        MigrationExecutor(connection).migrate(leaves)
    assert TextChunk.objects.filter(doc_id=doc.id).count() == 1
