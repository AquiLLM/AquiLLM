"""Database enforcement protects immutable provenance and safe rollback."""
from importlib import import_module
import pytest
from django.db import DatabaseError, connection, transaction
from apps.ingestion.models import IngestionBatch
from .output_support import upload

pytestmark = pytest.mark.django_db(transaction=True)


def test_database_rejects_manifest_identity_mutation(upload):
    from apps.ingestion.services.output_manifest import prepare_output_manifest
    item = upload()
    prepare_output_manifest(item.pk, using='default')
    with pytest.raises(DatabaseError), transaction.atomic(), connection.cursor() as cursor:
        cursor.execute('UPDATE aquillm_ingestionoutput SET element_key=%s WHERE item_id=%s', ['forged', item.pk])


def test_used_ledger_refuses_reverse(upload):
    from apps.ingestion.services.output_manifest import prepare_output_manifest
    from django.apps import apps
    item = upload()
    prepare_output_manifest(item.pk, using='default')
    migration = import_module('apps.ingestion.migrations.0004_ingestion_output_ledger_v1')
    with connection.schema_editor() as editor, pytest.raises(RuntimeError, match='provenance'):
        migration.refuse_used_reverse(apps, editor)


def test_unused_ledger_allows_reverse(upload):
    from django.apps import apps
    upload()
    migration = import_module('apps.ingestion.migrations.0004_ingestion_output_ledger_v1')
    with connection.schema_editor() as editor:
        migration.refuse_used_reverse(apps, editor)


def test_batch_cascade_removes_ledger_not_immutable_artifacts(upload):
    from apps.ingestion.services.output_manifest import prepare_output_manifest
    from apps.ingestion.models import IngestionOutput, SourceArtifact
    item = upload()
    prepare_output_manifest(item.pk, using='default')
    count = SourceArtifact.objects.count()
    IngestionBatch.objects.filter(pk=item.batch_id).delete()
    assert not IngestionOutput.objects.exists()
    assert SourceArtifact.objects.count() == count


def test_real_migration_reverse_refuses_committed_provenance(upload):
    from django.db.migrations.executor import MigrationExecutor
    from aquillm.task_ingest_uploaded import run_ingest_uploaded_file
    from apps.ingestion.models import IngestionOutput
    item = upload((('one.txt', b'body'),))
    run_ingest_uploaded_file(item.pk)
    item.refresh_from_db()
    expected = item.document_ids[:]
    executor = MigrationExecutor(connection)
    leaves = executor.loader.graph.leaf_nodes()
    try:
        with pytest.raises(RuntimeError, match='provenance'):
            executor.migrate([('apps_ingestion', '0003_source_artifacts_v1')])
    finally:
        MigrationExecutor(connection).migrate(leaves)
    item.refresh_from_db()
    assert item.document_ids == expected
    assert IngestionOutput.objects.get().document_id is not None


def test_real_unused_migration_round_trip_preserves_legacy(upload):
    from django.db.migrations.executor import MigrationExecutor
    item = upload()
    item.status = 'success'
    item.save(update_fields=['status'])
    executor = MigrationExecutor(connection)
    leaves = executor.loader.graph.leaf_nodes()
    try:
        executor.migrate([('apps_ingestion', '0003_source_artifacts_v1')])
    finally:
        MigrationExecutor(connection).migrate(leaves)
    item.refresh_from_db()
    assert item.status == 'success' and item.manifest_complete is False


def test_direct_orm_cannot_reopen_completed_output(upload):
    from aquillm.task_ingest_uploaded import run_ingest_uploaded_file
    from apps.ingestion.models import IngestionOutput
    item = upload((('one.txt', b'body'),))
    run_ingest_uploaded_file(item.pk)
    with pytest.raises(DatabaseError), transaction.atomic():
        IngestionOutput.objects.update(status='queued')


def test_ingestion_model_state_matches_allocated_migration():
    from django.core.management import call_command
    call_command('makemigrations', 'apps_ingestion', check=True, dry_run=True, verbosity=0)
