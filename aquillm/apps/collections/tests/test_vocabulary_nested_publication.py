"""Real publication composes with caller transactions without reopening history."""
from contextlib import ExitStack

import pytest
from django.db import IntegrityError, transaction

from apps.collections.models import VocabularyEntry, VocabularyVersion
from apps.collections.services.vocabulary_contracts import digest, version_payload
from apps.collections.tests.test_vocabulary_scope import entry, service, setup_scope

pytestmark = pytest.mark.django_db(transaction=True)
FIELDS = ("entry_key", "kind", "term", "expansions", "contributor")


def draft_case():
    actor, collection, _ = setup_scope("nested-publication")
    api = service()
    vocabulary = api.create_vocabulary(actor=actor, owner_collection=collection, name="Terms")
    draft = api.save_draft(actor=actor, vocabulary=vocabulary, expected_revision=1,
                          entries=[entry()])
    return actor, collection, vocabulary, draft


def assert_publication(version, draft, actor, collection):
    version.refresh_from_db()
    assert version.published_by_id == actor.pk
    assert version.owner_collection_id == collection.pk
    assert version.draft_revision == draft.revision
    rows = list(version.entries.order_by("entry_key").values(*FIELDS))
    assert rows == draft.entries
    assert set(version.entries.values_list("owner_collection_id", flat=True)) == {collection.pk}
    assert version.checksum == digest(version_payload(
        version.vocabulary_id, collection.pk, draft.revision, version.published_by_ref, rows))
    assert version.version_key == digest({"version": "vocabulary-key-v1", "checksum": version.checksum})


@pytest.mark.parametrize("outer_depth", [0, 1, 2])
def test_publish_inside_caller_atomic_retains_exact_entries_and_checksum(outer_depth):
    actor, collection, vocabulary, draft = draft_case()
    with ExitStack() as stack:
        for _ in range(outer_depth):
            stack.enter_context(transaction.atomic())
        version = service().publish_vocabulary(actor=actor, vocabulary=vocabulary,
                                               expected_revision=draft.revision)
        assert_publication(version, draft, actor, collection)
    assert_publication(version, draft, actor, collection)
    assert VocabularyVersion.objects.filter(vocabulary=vocabulary).count() == 1


@pytest.mark.parametrize("mutation", ["append", "update"])
def test_committed_publication_still_rejects_later_entry_mutation(mutation):
    actor, collection, vocabulary, draft = draft_case()
    version = service().publish_vocabulary(actor=actor, vocabulary=vocabulary,
                                           expected_revision=draft.revision)
    with pytest.raises(IntegrityError) as error:
        with transaction.atomic():
            if mutation == "append":
                VocabularyEntry.objects.create(version=version, owner_collection=collection,
                                                **entry(key="late-entry"))
            else:
                version.entries.update(term="changed after publication")
    assert error.value.__cause__.pgcode == "23514"
    assert_publication(version, draft, actor, collection)


def test_outer_rollback_leaves_no_partial_version_and_allows_clean_retry():
    actor, collection, vocabulary, draft = draft_case()
    with pytest.raises(RuntimeError, match="^rollback caller work$"):
        with transaction.atomic():
            version = service().publish_vocabulary(actor=actor, vocabulary=vocabulary,
                                                   expected_revision=draft.revision)
            assert_publication(version, draft, actor, collection)
            raise RuntimeError("rollback caller work")
    assert not VocabularyVersion.objects.filter(vocabulary=vocabulary).exists()
    assert not VocabularyEntry.objects.filter(owner_collection=collection).exists()
    version = service().publish_vocabulary(actor=actor, vocabulary=vocabulary,
                                           expected_revision=draft.revision)
    assert_publication(version, draft, actor, collection)
