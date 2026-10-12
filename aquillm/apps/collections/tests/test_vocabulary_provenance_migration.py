"""Fresh reversal and populated forward migration preserve closed publication history."""
import pytest
from django.db import IntegrityError, connection, transaction
from django.db.migrations.executor import MigrationExecutor
from django.db.migrations.exceptions import IrreversibleError

from apps.collections.models import VocabularyEntry, VocabularyVersion
from apps.collections.tests.test_vocabulary_nested_publication import (
    assert_publication, draft_case,
)
from apps.collections.tests.test_vocabulary_scope import entry, service
from apps.collections.tests.vocabulary_publication_test_support import version_values

pytestmark = pytest.mark.django_db(transaction=True)
OLD = ("apps_collections", "0004_vocabulary_versions_v1")
NEW = ("apps_collections", "0005_vocabulary_publication_transaction")


def columns():
    with connection.cursor() as cursor:
        return {column.name for column in connection.introspection.get_table_description(
            cursor, VocabularyVersion._meta.db_table)}


@pytest.mark.parametrize("populated", [False, True])
def test_provenance_migration_fresh_reverse_or_populated_closed_history(populated):
    executor = MigrationExecutor(connection)
    latest = executor.loader.graph.leaf_nodes()
    assert NEW in latest
    assert not VocabularyVersion.objects.exists()
    try:
        executor.migrate([OLD])
        assert "publication_transaction_id" not in columns()
        state = executor.loader.project_state([OLD]).apps
        old_version = state.get_model("apps_collections", "VocabularyVersion")
        old_entry = state.get_model("apps_collections", "VocabularyEntry")
        actor, collection, vocabulary, draft = draft_case()
        if populated:
            # Genuine v1 publication under its supported top-level transaction.
            with transaction.atomic():
                version = old_version.objects.create(
                    **version_values(actor, collection, vocabulary, draft))
                old_entry.objects.create(version_id=version.pk,
                                         owner_collection_id=collection.pk, **entry())
            before = old_version.objects.values().get(pk=version.pk)
            entry_before = list(old_entry.objects.filter(version_id=version.pk).values())
        MigrationExecutor(connection).migrate([NEW])
        assert "publication_transaction_id" in columns()
        if populated:
            version = VocabularyVersion.objects.get(pk=version.pk)
            assert version.publication_transaction_id == 0
            actual = VocabularyVersion.objects.values().get(pk=version.pk)
            assert actual.pop("publication_transaction_id") == 0
            assert actual == before
            assert list(VocabularyEntry.objects.filter(version_id=version.pk).values()) == entry_before
            assert_publication(version, draft, actor, collection)
            with pytest.raises(IntegrityError) as caught, transaction.atomic():
                VocabularyEntry.objects.create(version=version, owner_collection=collection,
                                               **entry(key="migration-append"))
            assert caught.value.__cause__.pgcode == "23514"
            with pytest.raises(IrreversibleError, match="transaction provenance"):
                MigrationExecutor(connection).migrate([OLD])
            assert "publication_transaction_id" in columns()
            assert_publication(version, draft, actor, collection)
        else:
            MigrationExecutor(connection).migrate([OLD])
            assert "publication_transaction_id" not in columns()
            MigrationExecutor(connection).migrate([NEW])
        # Both histories allow a fresh coherent nested publication after upgrade.
        if populated:
            draft = service().save_draft(actor=actor, vocabulary=vocabulary,
                                         expected_revision=draft.revision, entries=[entry()])
        with transaction.atomic():
            version = service().publish_vocabulary(actor=actor, vocabulary=vocabulary,
                                                   expected_revision=draft.revision)
            assert_publication(version, draft, actor, collection)
        assert_publication(version, draft, actor, collection)
    finally:
        MigrationExecutor(connection).migrate(latest)
