"""Database-owned top-level provenance supports subtransactions without reopening history."""
import json

import pytest
from django.db import IntegrityError, connection, connections, transaction

from apps.collections.models import VocabularyEntry, VocabularyVersion
from apps.collections.tests.test_vocabulary_nested_publication import (
    assert_publication, draft_case,
)
from apps.collections.tests.test_vocabulary_scope import entry, service
from apps.collections.tests.vocabulary_publication_test_support import (
    current_transaction, parent_xmin, version_values,
)

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.mark.parametrize("forged", [0, -17, "other_transaction"])
def test_insert_overwrites_client_provenance_and_released_savepoint_remains_owned(forged):
    actor, collection, vocabulary, draft = draft_case()
    with transaction.atomic():
        original = current_transaction()
        supplied = original + 1 if forged == "other_transaction" else forged
        with transaction.atomic():
            version = VocabularyVersion.objects.create(
                **version_values(actor, collection, vocabulary, draft),
                publication_transaction_id=supplied)
            version.refresh_from_db()
            assert version.publication_transaction_id == original
            assert parent_xmin(version) != original % 4294967296
        assert current_transaction() == original
        # The parent subtransaction has ended, but the original transaction still owns it.
        VocabularyEntry.objects.create(version=version, owner_collection=collection, **entry())
        assert_publication(version, draft, actor, collection)
    assert_publication(version, draft, actor, collection)
    with pytest.raises(IntegrityError) as caught, transaction.atomic():
        VocabularyEntry.objects.create(version=version, owner_collection=collection,
                                       **entry(key="committed-append"))
    assert caught.value.__cause__.pgcode == "23514"


@pytest.mark.parametrize("new_value", [0, "current_transaction"])
def test_committed_version_provenance_cannot_be_rewritten_to_reopen_history(new_value):
    actor, collection, vocabulary, draft = draft_case()
    version = service().publish_vocabulary(actor=actor, vocabulary=vocabulary,
                                           expected_revision=draft.revision)
    version.refresh_from_db()
    original = version.publication_transaction_id
    with pytest.raises(IntegrityError) as caught, transaction.atomic():
        current = current_transaction()
        assert current != original
        replacement = current if new_value == "current_transaction" else new_value
        VocabularyVersion.objects.filter(pk=version.pk).update(
            publication_transaction_id=replacement)
    assert caught.value.__cause__.pgcode == "23514"
    version.refresh_from_db()
    assert version.publication_transaction_id == original
    assert_publication(version, draft, actor, collection)


def test_other_backend_cannot_append_to_an_uncommitted_publication():
    actor, collection, vocabulary, draft = draft_case()
    alias = "vocabulary-provenance-other"
    original_wrapper = connections["default"]
    assert alias not in connections
    other = original_wrapper.copy(alias=alias)
    # PostgreSQL connection_created handlers resolve this exact alias.
    connections[alias] = other
    try:
        assert connections[alias] is other and connections["default"] is original_wrapper
        assert other.settings_dict == original_wrapper.settings_dict
        other.set_autocommit(True)
        with transaction.atomic():
            original = current_transaction()
            version = VocabularyVersion.objects.create(
                **version_values(actor, collection, vocabulary, draft))
            version.refresh_from_db()
            assert version.publication_transaction_id == original
            table = other.ops.quote_name(VocabularyVersion._meta.db_table)
            with other.cursor() as cursor:
                cursor.execute("SELECT txid_current(), pg_backend_pid()")
                foreign_transaction, foreign_pid = cursor.fetchone()
                assert foreign_transaction != original
                with connection.cursor() as own:
                    own.execute("SELECT pg_backend_pid()")
                    assert own.fetchone()[0] != foreign_pid
                cursor.execute(f"SELECT count(*) FROM {table} WHERE id = %s", [version.pk])
                assert cursor.fetchone() == (0,)
            table = other.ops.quote_name(VocabularyEntry._meta.db_table)
            with pytest.raises(IntegrityError) as caught, other.cursor() as cursor:
                cursor.execute(
                    f"INSERT INTO {table} (version_id, owner_collection_id, entry_key, kind, "
                    "term, expansions, contributor) VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s)",
                    [version.pk, collection.pk, "foreign", "alias", "test",
                     json.dumps(["foreign backend"]), "synthetic curator"])
            assert caught.value.__cause__.pgcode == "23514"
            VocabularyEntry.objects.create(version=version, owner_collection=collection, **entry())
            assert_publication(version, draft, actor, collection)
    finally:
        try:
            other.close()
        finally:
            del connections[alias]
    assert connections["default"] is original_wrapper
    assert_publication(version, draft, actor, collection)
