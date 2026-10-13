"""Real bounded source adapters preserve veto endpoints hidden by representatives."""
from dataclasses import replace
import pytest
from django.db import transaction

from apps.collections.models import CollectionPermission
from apps.collections.tests.test_vocabulary_scope import entry, publish, service
from apps.knowledge_graph.tests.test_build_orchestration_postgres_races import (
    _persist_active_ontology, _persist_document, _document_context, _ontology, _embedding_signature)


@pytest.mark.django_db(transaction=True)
def test_actual_source_memberships_survive_collection_and_canonical_adaptation(monkeypatch):
    from apps.knowledge_graph.services import builds
    from apps.knowledge_graph.models import (GraphArtifact, EntityMention, DocumentEntity,
        DocumentEntityMention, CollectionEntity, CollectionEntityDocumentLink)
    from apps.knowledge_graph.resolution.collection import (
        CollectionResolutionConfig, build_collection_snapshot, _load_resolution_source_rows,
        _snapshot_from_locked_manifest, resolve_collection_entities)
    from apps.knowledge_graph.resolution.canonical_snapshot_v2 import capture_source
    from apps.knowledge_graph.resolution.canonical import resolve_canonical_entities
    from apps.knowledge_graph.graph.filtering import FilterPolicy
    from apps.knowledge_graph.tests.test_collection_resolution import _session, _unit_vector
    from lib.knowledge_graph import config as configuration
    _persist_active_ontology()
    collection, document, chunk = _persist_document(label="hidden-veto-member")
    actor = document.ingested_by
    CollectionPermission.objects.create(collection=collection, user=actor, permission="MANAGE")
    _, version = publish(actor, collection, [entry("Orion", "MMLU", kind="never_merge")])
    service().attach_vocabulary(actor=actor, collection=collection, version=version, expected_revision=0)
    monkeypatch.setenv("KG_VOCABULARY_VETO_ENABLED", "1")
    settings = _document_context(document, chunk).settings
    monkeypatch.setattr(configuration, "load_extraction_settings", lambda: settings)
    context = builds._document_context(document.id, document.full_text_hash)
    artifact, _, _, _, _ = builds._bootstrap_document_build(context, builds.derive_document_build_key(context.identity))
    entities = []
    for index, (label, surface, start, end) in enumerate((("Orion", "MMLU", 11, 15), ("Orion", "Orion", 0, 5)), 1):
        mention = EntityMention.objects.create(artifact=artifact, document_id=document.id, chunk=chunk,
            start=start, end=end, raw_text=surface, normalized_text=surface.lower(), entity_type="model",
            extraction_confidence=0.9, position_basis="document_global")
        entity = DocumentEntity.objects.create(artifact=artifact, document_id=document.id,
            cluster_key=f"{index:064x}", label=label, normalized_label=label.lower(), entity_type="model",
            identifier="doi:10.1000/shared" if index == 1 else "")
        DocumentEntityMention.objects.create(document_entity=entity, mention=mention, method="root",
                                              resolver_version=artifact.resolver_version)
        entities.append(entity)
    artifact.status = "active"
    artifact.save(update_fields=["status"])
    config = CollectionResolutionConfig(vocabulary=context.vocabulary.policy)
    target, manifests = build_collection_snapshot(collection=collection, document_artifacts=(artifact,),
        ontology=_ontology(), extractor_version=artifact.extractor_version,
        resolver_version="collection-resolution-v1", filter_policy=FilterPolicy(),
        resolution_config=config, embedding_model_signature=_embedding_signature())
    with transaction.atomic():
        projected, relations = _load_resolution_source_rows(target, manifests, config=config)
        snapshot = _snapshot_from_locked_manifest(target, manifests)
    relay = next(e for e in projected if e.entity_id == entities[0].pk)
    assert relay.alias_evidence == () and relay.vocabulary_terms == ("mmlu",)
    session, _ = _session({"Orion": _unit_vector(1.0)})
    resolved = resolve_collection_entities(snapshot, projected, _ontology(), config=config,
        relations=relations, embedding_session=session)
    assert len(resolved.clusters) == 2
    for entity in entities:
        canonical = CollectionEntity.objects.create(artifact=target, collection=collection,
            cluster_key=entity.cluster_key, label=entity.label, normalized_label=entity.normalized_label,
            entity_type="model", identifier=entity.identifier, extraction_confidence=0.9,
            resolution_confidence=0.9, retrieval_utility=0.9, promotion_confidence=0.9)
        CollectionEntityDocumentLink.objects.create(artifact=target, manifest_input=manifests[0],
            document_entity=entity, collection_entity=canonical, score=0.9, method="exact",
            resolver_version=target.resolver_version, outcome="automatic", decision_checksum="6" * 64, reason="exact")
    target.status = "active"
    target.save(update_fields=["status"])
    with transaction.atomic():
        captured = capture_source(using="default", identity_policy="legacy")
    relay = next(e for e in captured.inputs if e.cluster_key == entities[0].cluster_key)
    assert relay.aliases == () and relay.vocabulary_terms == ("mmlu",)
    other = replace(next(e for e in captured.inputs if e.cluster_key == entities[1].cluster_key),
                    collection_id=collection.pk + 1000, entity_id=10000, identifier=relay.identifier)
    assert len(resolve_canonical_entities((relay, other), vocabulary=captured.vocabulary).components) == 2
    assert len(resolve_canonical_entities((relay, other)).components) == 1


@pytest.mark.django_db(transaction=True)
def test_no_applicable_veto_preserves_actual_canonical_snapshot_bytes(monkeypatch):
    from apps.knowledge_graph.tests.ready_subset_support import ScopeFixture
    from apps.knowledge_graph.resolution.canonical_snapshot_v2 import capture_source
    fixture = ScopeFixture(monkeypatch)
    fixture.persist_ready("A", with_evidence=True)
    monkeypatch.delenv("KG_VOCABULARY_VETO_ENABLED", raising=False)
    with transaction.atomic():
        disabled = capture_source(using="default", identity_policy="legacy")
    monkeypatch.setenv("KG_VOCABULARY_VETO_ENABLED", "1")
    with transaction.atomic():
        empty = capture_source(using="default", identity_policy="legacy")
    assert disabled.inputs and all(row.vocabulary_terms == () for row in disabled.inputs)
    assert disabled == empty and disabled.revision_digest == empty.revision_digest


def test_missing_required_memberships_and_footprint_overflow_are_unavailable():
    from types import SimpleNamespace
    from apps.collections.services.vocabulary_contracts import MAX_BYTES
    from apps.knowledge_graph.resolution.vocabulary_terms import decorate_canonical, terms_for
    from apps.knowledge_graph.resolution.vocabulary_capture import VocabularyGraphUnavailable
    from apps.knowledge_graph.tests.test_vocabulary_veto_resolvers import policy
    with pytest.raises(VocabularyGraphUnavailable):
        decorate_canonical((SimpleNamespace(entity_id=1),), (), {22: 1}, policy())
    with pytest.raises(VocabularyGraphUnavailable):
        terms_for(policy(), ("Alpha",), budget=[MAX_BYTES])
