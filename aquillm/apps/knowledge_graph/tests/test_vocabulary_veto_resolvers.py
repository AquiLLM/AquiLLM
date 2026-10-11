"""Explicit scoped vetoes bind actual resolver kernels and never grant merges."""
from dataclasses import replace

import pytest


def policy(scope=7):
    from apps.knowledge_graph.resolution.vocabulary_policy import ScopedVetoV1
    return ScopedVetoV1(((scope, (("alpha", "gamma"),)),), "a" * 64)


def test_document_veto_is_transitive_and_changes_result_identity():
    from apps.knowledge_graph.resolution.coreference import resolve_document_mentions
    from apps.knowledge_graph.tests.test_coreference import _mention, _ontology
    mentions = tuple(_mention(str(i), term, "model", start=i * 20,
        identifier="doi:10.1000/shared") for i, term in enumerate(("Alpha", "Beta", "Gamma")))
    baseline = resolve_document_mentions(mentions, _ontology(), identity_policy="contextual_v2")
    result = resolve_document_mentions(mentions, _ontology(), identity_policy="contextual_v2",
        vocabulary=policy(), collection_id=7)
    assert len(baseline.clusters) == 1
    assert all(not {"0", "2"} <= set(c.mention_ids) for c in result.clusters)
    assert result.checksum != baseline.checksum
    assert result.resolver_version == baseline.resolver_version


def test_negative_footprint_never_changes_unrelated_components_or_disabled_identity():
    from apps.knowledge_graph.resolution.canonical import CanonicalEntityInput, resolve_canonical_entities
    rows = tuple(CanonicalEntityInput(entity_id=i, collection_id=i, artifact_id=i,
        cluster_key=f"{i:064x}", label=label, normalized_label=label.lower(),
        entity_type="model", identifier="doi:10.1000/" + ("shared" if i < 4 else "other"),
        vocabulary_terms=("gamma",) if i == 3 else ())
        for i, label in enumerate(("Alpha", "Beta", "Relay", "Delta", "Epsilon"), 1))
    baseline = resolve_canonical_entities(tuple(replace(r, vocabulary_terms=()) for r in rows))
    disabled = resolve_canonical_entities(rows)
    assert disabled == baseline
    result = resolve_canonical_entities(rows, vocabulary=policy(1))
    assert all(not {1, 3} <= set(c.entity_ids) for c in result.components)
    assert any(c.entity_ids == (4, 5) for c in result.components)
    assert result.resolver_version == baseline.resolver_version


def test_collection_representative_cannot_hide_vetoed_membership():
    from apps.knowledge_graph.resolution.collection import (
        CollectionResolutionConfig, AliasEvidence, resolve_collection_entities)
    from apps.knowledge_graph.tests.test_collection_resolution import (
        _document_entity, _snapshot, _ontology, _session, _unit_vector)
    config = CollectionResolutionConfig(identity_policy="contextual_v2", vocabulary=policy())
    rows = (_document_entity(1, "Alpha", identifier="doi:10.1000/shared"),
        replace(_document_entity(2, "Beta", identifier="doi:10.1000/shared"),
                alias_evidence=(AliasEvidence("Gamma", "ontology_alias", 42),)))
    session, _ = _session({label: _unit_vector(1.0) for label in ("Alpha", "Beta")})
    result = resolve_collection_entities(_snapshot(config=config), rows, _ontology(),
                                         config=config, embedding_session=session)
    assert len(result.clusters) == 2


def test_canonical_representative_cannot_hide_vetoed_membership():
    from apps.knowledge_graph.resolution.canonical import (
        CanonicalEntityInput, CanonicalAliasEvidence, resolve_canonical_entities)
    rows = tuple(CanonicalEntityInput(entity_id=i, collection_id=i, artifact_id=i,
        cluster_key=f"{i:064x}", label=label, normalized_label=label.lower(),
        entity_type="model", identifier="doi:10.1000/shared")
        for i, label in enumerate(("Alpha", "Beta"), 1))
    rows = (rows[0], replace(rows[1], aliases=(
        CanonicalAliasEvidence("Gamma", "ontology_alias", 2, 42, 43),)))
    result = resolve_canonical_entities(rows, identity_policy="contextual_v2", vocabulary=policy(1))
    assert len(result.components) == 2


def test_collection_veto_overrides_shared_id_without_adding_alias_authority():
    from apps.knowledge_graph.resolution.collection import (
        CollectionResolutionConfig, resolve_collection_entities, resolution_config_checksum)
    from apps.knowledge_graph.tests.test_collection_resolution import _document_entity, _snapshot, _ontology
    base = CollectionResolutionConfig(identity_policy="contextual_v2")
    config = replace(base, vocabulary=policy())
    rows = tuple(_document_entity(i + 1, label, identifier="doi:10.1000/shared")
                 for i, label in enumerate(("Alpha", "Beta", "Gamma")))
    from apps.knowledge_graph.tests.test_collection_resolution import _session, _unit_vector
    session, _ = _session({label: _unit_vector(1.0) for label in ("Alpha", "Beta", "Gamma")})
    result = resolve_collection_entities(_snapshot(config=config), rows, _ontology(), config=config,
        embedding_session=session)
    assert all(not {1, 3} <= set(c.document_entity_ids) for c in result.clusters)
    assert resolution_config_checksum(config) != resolution_config_checksum(base)


def test_canonical_veto_uses_endpoint_membership_and_blocks_transitive_bridge():
    from apps.knowledge_graph.resolution.canonical import CanonicalEntityInput, resolve_canonical_entities
    rows = tuple(CanonicalEntityInput(entity_id=i, collection_id=i, artifact_id=i,
        cluster_key=f"{i:064x}", label=label, normalized_label=label.lower(),
        entity_type="model", identifier="doi:10.1000/shared")
        for i, label in enumerate(("Alpha", "Beta", "Gamma"), 1))
    baseline = resolve_canonical_entities(rows, identity_policy="contextual_v2")
    unrelated = resolve_canonical_entities(rows, identity_policy="contextual_v2", vocabulary=policy(9))
    result = resolve_canonical_entities(rows, identity_policy="contextual_v2", vocabulary=policy(1))
    assert len(baseline.components) == len(unrelated.components) == 1
    assert all(not {1, 3} <= set(c.entity_ids) for c in result.components)
    assert result.checksum != baseline.checksum
