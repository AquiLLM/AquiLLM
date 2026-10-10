"""Vocabulary management and frozen retrieval use actual database permissions."""

from importlib import import_module, util

import pytest
from django.contrib.auth.models import User

from apps.collections.models import Collection, CollectionPermission
from apps.collections.services.django_retrieval_authorization import (
    build_selected_scope_authorization_context,
)
from apps.documents.models import RawTextDocument


def service():
    name = "apps.collections.services.vocabulary"
    assert util.find_spec(name), "versioned vocabulary service is missing"
    return import_module(name)


def entry(term="μ_test", expansion="synthetic instance", key="entry-1", kind="alias"):
    return dict(entry_key=key, term=term, expansions=[expansion],
                contributor="synthetic curator", kind=kind)


def setup_scope(name="vocabulary", parent=None):
    actor = User.objects.create_user(username=name, is_staff=True)
    collection = Collection.objects.create(name=name, parent=parent)
    CollectionPermission.objects.create(user=actor, collection=collection, permission="MANAGE")
    document = RawTextDocument.objects.create(collection=collection, title="Synthetic", full_text="test", ingested_by=actor)
    auth = build_selected_scope_authorization_context(principal=actor,
        selected_collection_ids=(collection.pk,), selected_documents=[document])
    assert auth is not None
    return actor, collection, auth


def publish(actor, collection, entries=None):
    s = service()
    vocabulary = s.create_vocabulary(actor=actor, owner_collection=collection, name="Terms")
    draft = s.save_draft(actor=actor, vocabulary=vocabulary, expected_revision=1,
                         entries=entries or [entry()])
    version = s.publish_vocabulary(actor=actor, vocabulary=vocabulary,
                                   expected_revision=draft.revision)
    return vocabulary, version


@pytest.mark.django_db(transaction=True)
def test_draft_cas_permission_duplicate_publish_and_explicit_pin():
    s = service()
    actor, collection, auth = setup_scope()
    vocabulary, first = publish(actor, collection)
    with pytest.raises(ValueError):
        s.publish_vocabulary(actor=actor, vocabulary=vocabulary, expected_revision=2)
    with pytest.raises(ValueError):
        s.save_draft(actor=actor, vocabulary=vocabulary, expected_revision=1, entries=[entry()])
    outsider = User.objects.create_user(username="outsider")
    with pytest.raises(PermissionError):
        s.save_draft(actor=outsider, vocabulary=vocabulary, expected_revision=2, entries=[entry()])
    attachment = s.attach_vocabulary(actor=actor, collection=collection, version=first,
                                     expected_revision=0)
    snapshot = s.freeze_vocabulary(authorization=auth, selected_collection_ids=(collection.pk,))
    draft = s.save_draft(actor=actor, vocabulary=vocabulary, expected_revision=2,
                         entries=[entry(expansion="new instance")])
    second = s.publish_vocabulary(actor=actor, vocabulary=vocabulary,
                                    expected_revision=draft.revision)
    assert snapshot.version_keys == (first.version_key,)
    assert s.validate_vocabulary(snapshot, authorization=auth)
    s.attach_vocabulary(actor=actor, collection=collection, version=second,
                         expected_revision=attachment.revision)
    with pytest.raises(ValueError):
        s.validate_vocabulary(snapshot, authorization=auth)


@pytest.mark.django_db(transaction=True)
def test_ancestor_inheritance_needs_explicit_attachment_and_current_source_access():
    s = service()
    actor, parent, _ = setup_scope()
    child = Collection.objects.create(name="selected child", parent=parent)
    CollectionPermission.objects.create(user=actor, collection=child, permission="VIEW")
    doc = RawTextDocument.objects.create(collection=child, title="child", full_text="test", ingested_by=actor)
    auth = build_selected_scope_authorization_context(principal=actor,
        selected_collection_ids=(child.pk,), selected_documents=[doc])
    _, version = publish(actor, parent)
    pin = s.attach_vocabulary(actor=actor, collection=parent, version=version,
                              expected_revision=0, inherit_descendants=False)
    assert s.freeze_vocabulary(authorization=auth,
        selected_collection_ids=(child.pk,)).aliases == ()
    s.attach_vocabulary(actor=actor, collection=parent, version=version,
                         expected_revision=pin.revision, inherit_descendants=True)
    frozen = s.freeze_vocabulary(authorization=auth, selected_collection_ids=(child.pk,))
    assert frozen.aliases[0].term == "μ_test"
    CollectionPermission.objects.filter(user=actor, collection=parent).delete()
    with pytest.raises((PermissionError, ValueError)):
        s.validate_vocabulary(frozen, authorization=auth)


@pytest.mark.django_db(transaction=True)
def test_unselected_vocabulary_never_leaks_and_detachment_revokes():
    s = service()
    actor, selected, auth = setup_scope()
    foreign = Collection.objects.create(name="unselected")
    CollectionPermission.objects.create(user=actor, collection=foreign, permission="MANAGE")
    _, private = publish(actor, foreign)
    s.attach_vocabulary(actor=actor, collection=foreign, version=private, expected_revision=0)
    frozen = s.freeze_vocabulary(authorization=auth, selected_collection_ids=(selected.pk,))
    assert frozen.aliases == ()
    _, local = publish(actor, selected)
    pin = s.attach_vocabulary(actor=actor, collection=selected, version=local, expected_revision=0)
    frozen = s.freeze_vocabulary(authorization=auth, selected_collection_ids=(selected.pk,))
    pin.delete()
    with pytest.raises(ValueError):
        s.validate_vocabulary(frozen, authorization=auth)


@pytest.mark.django_db(transaction=True)
def test_conflicting_entry_and_alias_cannot_override_never_merge():
    s = service()
    actor, collection, _ = setup_scope()
    vocabulary = s.create_vocabulary(actor=actor, owner_collection=collection, name="Terms")
    with pytest.raises(ValueError):
        s.save_draft(actor=actor, vocabulary=vocabulary, expected_revision=1, entries=[
            entry("A", "B", "alias"), entry("A", "B", "veto", "never_merge")])
