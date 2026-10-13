"""Persisted vocabulary pins enter graph identity and remain revocable."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from apps.collections.models import Collection, CollectionPermission
from apps.collections.tests.test_vocabulary_scope import setup_scope, publish, service, entry
from apps.documents.services.source_loading import SourceRuntime, source_runtime_scope
from lib.retrieval.turn_budget import TurnBudget, TurnLimits


def api():
    from apps.knowledge_graph.resolution import vocabulary_capture
    return vocabulary_capture


def setup(monkeypatch):
    monkeypatch.setenv("KG_VOCABULARY_VETO_ENABLED", "1")
    actor, collection, authorization = setup_scope()
    vocabulary, version = publish(actor, collection, [entry("Alpha", "Gamma", kind="never_merge")])
    pin = service().attach_vocabulary(actor=actor, collection=collection, version=version, expected_revision=0)
    return actor, collection, authorization, vocabulary, version, pin


@pytest.mark.django_db(transaction=True)
def test_actual_document_identity_artifact_and_frozen_consumer_reject_repin(monkeypatch):
    actor, collection, auth, vocabulary, version, pin = setup(monkeypatch)
    v = api()
    from apps.knowledge_graph.extraction.pipeline import document_artifact_identity_values
    from apps.knowledge_graph.models import GraphArtifact, graph_identity_checksum
    capture = v.capture_build_vocabulary((collection.pk,))
    capsule = capture.get(collection.pk)
    settings = SimpleNamespace(provider="synthetic", model_id="fixture", model_revision="1")
    values = document_artifact_identity_values(next(iter(auth.selected_document_ids)), "a" * 64,
        "test", "b" * 64, settings=settings, identity_policy="contextual_v2", vocabulary=capsule.policy)
    base = graph_identity_checksum("document-resolver", values["resolver_version"])
    artifact = GraphArtifact.objects.create(**values, metadata=v.metadata(capsule, base))
    assert values["resolution_config_checksum"] != base
    with pytest.raises(v.VocabularyGraphUnavailable):
        v.capture_retrieval_vocabulary(auth)
    runtime = SourceRuntime(TurnBudget(TurnLimits()), auth)
    with source_runtime_scope(runtime):
        frozen = v.capture_retrieval_vocabulary(auth)
        assert v.validate_artifact(artifact, frozen, collection_id=collection.pk) == capsule.policy
        draft = service().save_draft(actor=actor, vocabulary=vocabulary, expected_revision=2,
            entries=[entry("Alpha", "Beta", kind="never_merge")])
        second = service().publish_vocabulary(actor=actor, vocabulary=vocabulary, expected_revision=draft.revision)
        service().attach_vocabulary(actor=actor, collection=collection, version=second, expected_revision=pin.revision)
        with pytest.raises(v.VocabularyGraphUnavailable):
            v.capture_retrieval_vocabulary(auth)
    current = v.capture_build_vocabulary((collection.pk,))
    with pytest.raises(v.VocabularyGraphUnavailable):
        v.validate_artifact(artifact, current, collection_id=collection.pk)


@pytest.mark.django_db(transaction=True)
@pytest.mark.parametrize("change", ["detach", "move", "permission"])
def test_actual_current_policy_changes_revoke_captured_use(monkeypatch, change):
    actor, collection, auth, _, _, pin = setup(monkeypatch)
    v = api()
    runtime = SourceRuntime(TurnBudget(TurnLimits()), auth)
    with source_runtime_scope(runtime):
        v.capture_retrieval_vocabulary(auth)
        if change == "detach":
            pin.delete()
        elif change == "move":
            collection.parent = Collection.objects.create(name="new parent")
            collection.save(update_fields=["parent"])
        else:
            CollectionPermission.objects.filter(user=actor, collection=collection).delete()
        with pytest.raises(v.VocabularyGraphUnavailable):
            v.capture_retrieval_vocabulary(auth)


def test_disabled_capture_has_no_storage_work(monkeypatch):
    v = api()
    monkeypatch.delenv("KG_VOCABULARY_VETO_ENABLED", raising=False)
    monkeypatch.setattr(v, "read_pins", lambda *a, **k: pytest.fail("disabled storage read"))
    assert v.capture_build_vocabulary((7,)) is None
    assert v.capture_retrieval_vocabulary(None) is None


@pytest.mark.django_db(transaction=True)
def test_graph_ancestor_requires_current_vocabulary_source_access(monkeypatch):
    from apps.collections.services.django_retrieval_authorization import build_selected_scope_authorization_context
    from apps.documents.models import RawTextDocument
    actor, parent, _, _, version, pin = setup(monkeypatch)
    child = Collection.objects.create(name="graph selected child", parent=parent)
    CollectionPermission.objects.create(user=actor, collection=child, permission="VIEW")
    doc = RawTextDocument.objects.create(collection=child, title="child", full_text="test", ingested_by=actor)
    auth = build_selected_scope_authorization_context(principal=actor,
        selected_collection_ids=(child.pk,), selected_documents=[doc])
    service().attach_vocabulary(actor=actor, collection=parent, version=version,
        expected_revision=pin.revision, inherit_descendants=True)
    runtime = SourceRuntime(TurnBudget(TurnLimits()), auth)
    with source_runtime_scope(runtime):
        captured = api().capture_retrieval_vocabulary(auth)
        assert captured.get(child.pk).policy.scopes[0][0] == child.pk
        CollectionPermission.objects.filter(user=actor, collection=parent).delete()
        with pytest.raises(api().VocabularyGraphUnavailable):
            api().capture_retrieval_vocabulary(auth)


@pytest.mark.django_db(transaction=True)
def test_global_persisted_entry_cap_fails_unavailable_without_partial_veto(monkeypatch):
    monkeypatch.setenv("KG_VOCABULARY_VETO_ENABLED", "1")
    actor, collection, _ = setup_scope()
    for prefix in ("left", "right"):
        vocabulary = service().create_vocabulary(actor=actor, owner_collection=collection, name=prefix)
        draft = service().save_draft(actor=actor, vocabulary=vocabulary, expected_revision=1, entries=[
            entry(prefix + str(i), "other" + str(i), key=str(i), kind="never_merge")
            for i in range(129)])
        version = service().publish_vocabulary(actor=actor, vocabulary=vocabulary, expected_revision=draft.revision)
        service().attach_vocabulary(actor=actor, collection=collection, version=version, expected_revision=0)
    with pytest.raises(api().VocabularyGraphUnavailable):
        api().capture_build_vocabulary((collection.pk,))
