"""Authenticated revision-CAS vocabulary authoring; publication never repins."""
from django.db import transaction
from django.utils.crypto import salted_hmac

from apps.collections.models.vocabulary import (
    VocabularyAttachment, VocabularyDraft, VocabularyEntry, VocabularySet, VocabularyVersion,
)
from .vocabulary_access import require_permission
from .vocabulary_contracts import digest, text, validate_entries, version_payload


def create_vocabulary(*, actor, owner_collection, name, using="default"):
    with transaction.atomic(using=using):
        require_permission(actor, owner_collection.pk, using, "MANAGE")
        vocabulary = VocabularySet.objects.using(using).create(owner_collection=owner_collection, name=text(name, 100))
        VocabularyDraft.objects.using(using).create(vocabulary=vocabulary, updated_by=actor)
        return vocabulary


def _locked(vocabulary, actor, using):
    value = VocabularySet.objects.using(using).select_for_update().get(pk=vocabulary.pk)
    require_permission(actor, value.owner_collection_id, using, "MANAGE")
    return value, VocabularyDraft.objects.using(using).select_for_update().get(vocabulary=value)


def save_draft(*, actor, vocabulary, expected_revision, entries, using="default"):
    rows = validate_entries(entries)
    with transaction.atomic(using=using):
        value, draft = _locked(vocabulary, actor, using)
        if type(expected_revision) is not int or draft.revision != expected_revision:
            raise ValueError("stale vocabulary draft")
        draft.entries, draft.revision, draft.updated_by = list(rows), draft.revision + 1, actor
        draft.save(using=using, update_fields=("entries", "revision", "updated_by"))
        return draft


def publish_vocabulary(*, actor, vocabulary, expected_revision, using="default"):
    with transaction.atomic(using=using):
        value, draft = _locked(vocabulary, actor, using)
        if type(expected_revision) is not int or draft.revision != expected_revision:
            raise ValueError("stale vocabulary draft")
        if VocabularyVersion.objects.using(using).filter(vocabulary=value, draft_revision=draft.revision).exists():
            raise ValueError("duplicate vocabulary publication")
        rows = validate_entries(draft.entries)
        actor_ref = salted_hmac("vocabulary-publisher-v1", f"{using}:{actor.pk}", algorithm="sha256").hexdigest()
        checksum = digest(version_payload(value.pk, value.owner_collection_id, draft.revision, actor_ref, rows))
        version = VocabularyVersion.objects.using(using).create(
            vocabulary=value, owner_collection_id=value.owner_collection_id,
            version_key=digest({"version": "vocabulary-key-v1", "checksum": checksum}),
            draft_revision=draft.revision, checksum=checksum, published_by=actor, published_by_ref=actor_ref)
        VocabularyEntry.objects.using(using).bulk_create([
            VocabularyEntry(version=version, owner_collection_id=value.owner_collection_id, **row)
            for row in rows
        ])
        return version


def attach_vocabulary(*, actor, collection, version, expected_revision, inherit_descendants=False, using="default"):
    if type(expected_revision) is not int or type(inherit_descendants) is not bool:
        raise ValueError("invalid vocabulary pin revision")
    with transaction.atomic(using=using):
        version = VocabularyVersion.objects.using(using).select_related("vocabulary").get(pk=version.pk)
        require_permission(actor, version.owner_collection_id, using, "MANAGE")
        require_permission(actor, collection.pk, using, "MANAGE")
        # Serialize the absent-row case as well as updates.
        VocabularySet.objects.using(using).select_for_update().get(pk=version.vocabulary_id)
        row = VocabularyAttachment.objects.using(using).select_for_update().filter(
            collection=collection, vocabulary_id=version.vocabulary_id).first()
        if (row.revision if row else 0) != expected_revision:
            raise ValueError("stale vocabulary attachment")
        if row is None:
            return VocabularyAttachment.objects.using(using).create(
                collection=collection, vocabulary_id=version.vocabulary_id,
                pinned_version=version, inherit_descendants=inherit_descendants)
        row.pinned_version, row.inherit_descendants = version, inherit_descendants
        row.revision += 1
        row.save(using=using, update_fields=("pinned_version", "inherit_descendants", "revision"))
        return row


def detach_vocabulary(*, actor, attachment, expected_revision, using="default"):
    with transaction.atomic(using=using):
        row = VocabularyAttachment.objects.using(using).select_for_update().get(pk=attachment.pk)
        require_permission(actor, row.collection_id, using, "MANAGE")
        if type(expected_revision) is not int or row.revision != expected_revision:
            raise ValueError("stale vocabulary attachment")
        row.delete(using=using)
