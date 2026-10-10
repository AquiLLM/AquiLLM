"""Additive recovery schema permits old-image inserts without triggering work."""
from uuid import uuid4
import pytest
from django.db import connection
from django.db.migrations.executor import MigrationExecutor


@pytest.mark.django_db(transaction=True)
def test_failure_schema_preserves_old_rows_and_old_orm_inserts():
    before = [("apps_documents", "0007_textchunk_embedding_provenance")]
    after = [("apps_documents", "0008_chunkpublication_failure_fence")]
    executor = MigrationExecutor(connection)
    leaves = executor.loader.graph.leaf_nodes()
    try:
        executor.migrate(before)
        old_model = executor.loader.project_state(before).apps.get_model("apps_documents", "ChunkPublication")
        fields = dict(concrete_model_label="apps_documents.rawtextdocument", document_pkid=1,
                      document_id=uuid4(), source_hash="a" * 64, collection_ids=[1])
        old = old_model.objects.create(**fields)
        MigrationExecutor(connection).migrate(after)
        current = MigrationExecutor(connection).loader.project_state(after).apps.get_model("apps_documents", "ChunkPublication")
        assert current.objects.get(pk=old.pk).generation is None
        assert current.objects.get(pk=old.pk).failure_kind is None
        fields["source_hash"] = "b" * 64
        inserted = old_model.objects.create(**fields)
        assert current.objects.get(pk=inserted.pk).generation is None
        assert current.objects.get(pk=inserted.pk).failure_kind is None
        assert current.objects.count() == 2
    finally:
        MigrationExecutor(connection).migrate(leaves)
