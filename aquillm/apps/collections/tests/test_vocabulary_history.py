"""Immutable publication history permits only exact owner and actor deletion."""

import pytest
from django.db import IntegrityError, connection, transaction

from apps.collections.models import Collection
from apps.collections.tests.test_vocabulary_scope import publish, service, setup_scope


@pytest.mark.django_db(transaction=True)
@pytest.mark.parametrize("queryset", [False, True])
def test_exact_collection_deletion_cascades_published_history(queryset):
    s = service()
    actor, collection, _ = setup_scope()
    vocabulary, version = publish(actor, collection)
    s.attach_vocabulary(actor=actor, collection=collection, version=version, expected_revision=0)
    version_pk = version.pk
    if queryset:
        Collection.objects.filter(pk=collection.pk).delete()
    else:
        collection.delete()
    assert not type(version).objects.filter(pk=version_pk).exists()
    assert not type(vocabulary).objects.filter(pk=vocabulary.pk).exists()


@pytest.mark.django_db(transaction=True)
def test_raw_mutation_append_and_partial_history_delete_are_refused():
    service()
    actor, collection, _ = setup_scope()
    vocabulary, version = publish(actor, collection)
    from apps.collections.models.vocabulary import VocabularyEntry

    row = VocabularyEntry.objects.get(version=version)
    for model, pk, assignment in (
        (type(version), version.pk, "checksum = '" + "b" * 64 + "'"),
        (type(vocabulary), vocabulary.pk, "owner_collection_id = 999999"),
        (VocabularyEntry, row.pk, "term = 'mutated'"),
    ):
        table = connection.ops.quote_name(model._meta.db_table)
        with pytest.raises(IntegrityError), transaction.atomic(), connection.cursor() as cursor:
            cursor.execute(f"UPDATE {table} SET {assignment} WHERE id = %s", [pk])
    for model, pk in ((VocabularyEntry, row.pk), (type(version), version.pk),
                      (type(vocabulary), vocabulary.pk)):
        with pytest.raises((IntegrityError, ValueError)), transaction.atomic():
            model.objects.filter(pk=pk).delete()
    with pytest.raises(IntegrityError), transaction.atomic():
        VocabularyEntry.objects.create(version=version, owner_collection=collection,
            entry_key="late", kind="alias", term="late", expansions=["append"], contributor="actor")
    assert VocabularyEntry.objects.filter(version=version).count() == 1


@pytest.mark.django_db(transaction=True)
def test_actor_deletion_preserves_immutable_attribution_without_blocking():
    service()
    actor, collection, _ = setup_scope()
    _, version = publish(actor, collection)
    attribution, checksum = version.published_by_ref, version.checksum
    with pytest.raises(IntegrityError), transaction.atomic():
        type(version).objects.filter(pk=version.pk).update(published_by=None)
    from apps.documents.models import RawTextDocument
    RawTextDocument.objects.filter(ingested_by=actor).delete()
    actor.delete()
    version.refresh_from_db()
    assert version.published_by_id is None
    assert version.published_by_ref == attribution
    assert version.checksum == checksum


@pytest.mark.django_db(transaction=True)
def test_draft_and_attachment_owner_links_cannot_be_reassigned():
    s = service()
    actor, collection, _ = setup_scope()
    vocabulary, version = publish(actor, collection)
    other = s.create_vocabulary(actor=actor, owner_collection=collection, name="other")
    attachment = s.attach_vocabulary(actor=actor, collection=collection, version=version, expected_revision=0)
    from apps.collections.models.vocabulary import VocabularyDraft
    VocabularyDraft.objects.filter(vocabulary=other).delete()
    with pytest.raises(IntegrityError), transaction.atomic():
        VocabularyDraft.objects.filter(vocabulary=vocabulary).update(vocabulary=other)
    with pytest.raises(IntegrityError), transaction.atomic():
        type(attachment).objects.filter(pk=attachment.pk).update(collection=Collection.objects.create(name="elsewhere"))


@pytest.mark.django_db(transaction=True)
def test_collection_lifecycle_bound_counts_vocabulary_history(monkeypatch):
    service()
    actor, collection, _ = setup_scope()
    publish(actor, collection)
    from apps.collections.models import CollectionPermission
    from apps.documents.models import RawTextDocument
    RawTextDocument.objects.filter(collection=collection).delete()
    CollectionPermission.objects.filter(collection=collection).delete()
    from apps.knowledge_graph.graph import invalidation

    monkeypatch.setattr(invalidation, "_MAX_LIFECYCLE_ROWS", 4)
    with pytest.raises(RuntimeError, match="bounded lifecycle"):
        collection.delete()
    assert Collection.objects.filter(pk=collection.pk).exists()
    monkeypatch.setattr(invalidation, "_MAX_LIFECYCLE_ROWS", 5)
    collection.delete()
    assert not Collection.objects.filter(pk=collection.pk).exists()
