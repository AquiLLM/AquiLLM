"""Actual local producers bind immutable vocabulary without provider dispatch."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
from django.db import transaction

from apps.collections.models import CollectionPermission
from apps.collections.tests.test_vocabulary_scope import entry, publish, service
from apps.knowledge_graph.tests.test_build_orchestration_postgres_races import (
    _persist_active_ontology, _persist_document, _document_context, _ontology, _embedding_signature,
)


@pytest.mark.django_db(transaction=True)
def test_actual_document_bootstrap_collection_snapshot_and_canonical_capture(monkeypatch):
    from apps.knowledge_graph.services import builds
    from apps.knowledge_graph.resolution import vocabulary_capture as v
    from apps.knowledge_graph.resolution.collection import (
        CollectionResolutionConfig, build_collection_snapshot, resolution_config_checksum)
    from apps.knowledge_graph.graph.filtering import FilterPolicy
    from apps.knowledge_graph.models import GraphArtifact
    from apps.knowledge_graph.resolution.canonical_snapshot_v2 import capture_source
    from lib.knowledge_graph import config as configuration
    _persist_active_ontology()
    collection, document, chunk = _persist_document(label="vocabulary-producer")
    actor = document.ingested_by
    CollectionPermission.objects.create(collection=collection, user=actor, permission="MANAGE")
    settings = _document_context(document, chunk).settings
    monkeypatch.setattr(configuration, "load_extraction_settings", lambda: settings)
    baseline = builds._document_context(document.id, document.full_text_hash)
    _, version = publish(actor, collection, [entry("Alpha", "Gamma", kind="never_merge")])
    pin = service().attach_vocabulary(actor=actor, collection=collection, version=version, expected_revision=0)
    monkeypatch.setenv("KG_VOCABULARY_VETO_ENABLED", "1")
    context = builds._document_context(document.id, document.full_text_hash)
    assert context.identity.resolver_checksum != baseline.identity.resolver_checksum
    assert builds.derive_document_build_key(context.identity) != builds.derive_document_build_key(baseline.identity)
    artifact, _, _, _, _ = builds._bootstrap_document_build(
        context, builds.derive_document_build_key(context.identity))
    capture = v.capture_build_vocabulary((collection.pk,))
    assert v.validate_artifact(artifact, capture, collection_id=collection.pk) == context.vocabulary.policy
    artifact.status = GraphArtifact.Status.ACTIVE
    artifact.save(update_fields=["status"])
    config = CollectionResolutionConfig()
    collection_artifact, _ = build_collection_snapshot(collection=collection, document_artifacts=(artifact,),
        ontology=_ontology(), extractor_version=artifact.extractor_version,
        resolver_version="collection-resolution-v1", filter_policy=FilterPolicy(),
        resolution_config=config, embedding_model_signature=_embedding_signature())
    assert collection_artifact.resolution_config_checksum != resolution_config_checksum(config)
    collection_artifact.status = GraphArtifact.Status.ACTIVE
    collection_artifact.save(update_fields=["status"])
    with transaction.atomic():
        snapshot = capture_source(using="default", identity_policy="legacy")
    assert snapshot.vocabulary == capture.policy
    pin.delete()
    with transaction.atomic(), pytest.raises(v.VocabularyGraphUnavailable):
        capture_source(using="default", identity_policy="legacy")
