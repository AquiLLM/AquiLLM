"""Actual migration reversal, typed owners, and aggregate capture limits."""
import pytest
from django.db import connection, transaction, IntegrityError
from django.db.migrations.executor import MigrationExecutor
from django.db.migrations.exceptions import IrreversibleError

from apps.collections.models import Collection, VocabularyVersion, VocabularyEntry
from apps.collections.tests.test_vocabulary_scope import setup_scope, publish, service, entry


@pytest.mark.django_db(transaction=True)
def test_used_migration_refuses_reversal_without_losing_history():
    actor, collection, _ = setup_scope()
    _, version = publish(actor, collection)
    with pytest.raises(IrreversibleError):
        MigrationExecutor(connection).migrate([
            ("apps_collections", "0003_schema_head_and_generation_fencing")])
    assert VocabularyVersion.objects.get(pk=version.pk).checksum == version.checksum


@pytest.mark.django_db(transaction=True)
def test_wrong_owner_insert_and_unrelated_collection_delete_cannot_erase_history():
    actor, collection, _ = setup_scope()
    vocabulary, version = publish(actor, collection)
    foreign = Collection.objects.create(name="wrong owner")
    foreign_pk = foreign.pk
    with pytest.raises(IntegrityError), transaction.atomic():
        VocabularyVersion.objects.create(vocabulary=vocabulary, owner_collection=foreign,
            version_key="b" * 64, draft_revision=99, checksum="b" * 64,
            published_by=actor, published_by_ref="b" * 64)
    with pytest.raises(IntegrityError), transaction.atomic():
        VocabularyEntry.objects.filter(version=version).delete()
        foreign.delete()
    assert VocabularyEntry.objects.filter(version=version).exists()
    assert Collection.objects.filter(pk=foreign_pk).exists()


@pytest.mark.django_db(transaction=True)
def test_global_entry_bound_rejects_instead_of_dropping_vetoes():
    actor, collection, authorization = setup_scope()
    s = service()
    for part in range(2):
        vocabulary = s.create_vocabulary(actor=actor, owner_collection=collection, name=f"Terms {part}")
        draft = s.save_draft(actor=actor, vocabulary=vocabulary, expected_revision=1,
            entries=[entry(f"term_{part}_{i}", f"separate_{part}_{i}", f"e{i}", "never_merge") for i in range(129)])
        version = s.publish_vocabulary(actor=actor, vocabulary=vocabulary, expected_revision=draft.revision)
        s.attach_vocabulary(actor=actor, collection=collection, version=version, expected_revision=0)
    with pytest.raises(ValueError, match="aggregate vocabulary limit"):
        s.freeze_vocabulary(authorization=authorization, selected_collection_ids=(collection.pk,))


@pytest.mark.django_db(transaction=True)
def test_collection_move_revokes_frozen_ancestor_path():
    actor, collection, authorization = setup_scope()
    _, version = publish(actor, collection)
    service().attach_vocabulary(actor=actor, collection=collection, version=version, expected_revision=0)
    frozen = service().freeze_vocabulary(authorization=authorization,
        selected_collection_ids=(collection.pk,))
    new_parent = Collection.objects.create(name="new parent")
    collection.move_to(new_parent)
    with pytest.raises(ValueError):
        service().validate_vocabulary(frozen, authorization=authorization)
