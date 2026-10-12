"""Contextual graph identity regression controls extracted from legacy suites."""
import pytest


@pytest.mark.django_db
def test_document_context_binds_effective_contextual_policy(monkeypatch):
    from test_coreference import _ontology

    from apps.knowledge_graph.services import builds

    ontology = _ontology()
    ontology.version = "1.0.0"

    from apps.knowledge_graph.tests.legacy_preflight_support import legacy_owner
    document = legacy_owner(monkeypatch)
    monkeypatch.setattr(
        builds, "_ontology_activation_signature", lambda *_args: "c" * 64
    )
    monkeypatch.setenv("KG_IDENTITY_POLICY", "legacy")
    legacy = builds._document_context(document.id, document.full_text_hash, ontology=ontology)
    monkeypatch.setenv("KG_IDENTITY_POLICY", "contextual_v2")
    contextual = builds._document_context(document.id, document.full_text_hash, ontology=ontology)
    assert contextual.identity.resolver_version == "document-coreference-v4-contextual"
    assert contextual.identity.resolver_checksum != legacy.identity.resolver_checksum
    assert builds.derive_document_build_key(
        contextual.identity
    ) != builds.derive_document_build_key(legacy.identity)
    values = builds._document_artifact_values(
        contextual, builds.derive_document_build_key(contextual.identity)
    )
    assert values["resolver_version"] == contextual.identity.resolver_version
    assert values["resolution_config_checksum"] == contextual.identity.resolver_checksum


def test_task9_and_filter_markers_persist_every_checksum_addressed_row_cap():
    import inspect
    from apps.knowledge_graph.graph import assembly, filtering
    from apps.knowledge_graph.resolution import collection as collection_resolution

    persist_source = inspect.getsource(
        collection_resolution.persist_collection_resolution
    )
    marker_validation_source = inspect.getsource(
        collection_resolution._collection_resolution_marker_is_valid
    )
    filter_source = inspect.getsource(filtering.create_filter_rerun_artifact)
    lineage_source = inspect.getsource(assembly._validate_task9_lineage_node)
    from apps.knowledge_graph.resolution.vocabulary_builds import collection_replay_config
    assert "config=collection_replay_config(artifact, marker)" in lineage_source
    replay_config_source = inspect.getsource(collection_replay_config)

    for field_name in (
        "max_entities",
        "max_memberships",
        "max_relations",
        "max_links",
    ):
        marker_field = f'"{field_name}"'
        assert marker_field in persist_source
        assert marker_field in marker_validation_source
        assert marker_field in filter_source
        assert marker_field in replay_config_source
