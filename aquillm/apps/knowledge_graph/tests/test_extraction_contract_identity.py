from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from uuid import UUID

import pytest

from lib.knowledge_graph.config import load_extraction_settings


def test_document_artifact_identity_uses_shared_contract():
    from apps.knowledge_graph.extraction.pipeline import (
        document_artifact_identity_values,
    )
    from lib.knowledge_graph.extraction_identity import extraction_identity

    settings = load_extraction_settings({})
    arguments = (
        UUID("11111111-1111-4111-8111-111111111111"),
        "a" * 64,
        "1.0",
        "b" * 64,
    )
    before = document_artifact_identity_values(*arguments, settings=settings)
    after = document_artifact_identity_values(
        *arguments, settings=replace(settings, batch_size=2)
    )
    assert before["extractor_version"] == extraction_identity(settings)
    assert before["extractor_version"] != after["extractor_version"]
    assert {
        key: value for key, value in before.items() if key != "extractor_version"
    } == {key: value for key, value in after.items() if key != "extractor_version"}


def test_document_build_key_changes_when_adapter_contract_changes(monkeypatch):
    import lib.knowledge_graph.extraction_identity as identity_module
    from apps.knowledge_graph.extraction import pipeline
    from apps.knowledge_graph.projection import representation_inputs
    from apps.knowledge_graph.services import builds

    document_id = UUID("11111111-1111-4111-8111-111111111111")
    document = SimpleNamespace(
        collection_id=17,
        _meta=SimpleNamespace(label_lower="apps_documents.rawtextdocument"),
    )
    monkeypatch.setattr(
        pipeline, "_get_concrete_document", lambda *args, **kwargs: document
    )
    monkeypatch.setattr(pipeline, "_validate_source", lambda *args: None)
    monkeypatch.setattr(pipeline, "_ordered_chunks", lambda *args, **kwargs: ())
    monkeypatch.setattr(
        representation_inputs, "capture_build_snapshot", lambda *args: None
    )
    monkeypatch.setattr(builds, "document_capsule", lambda *args: None)
    monkeypatch.setattr(
        builds, "_ontology_activation_signature", lambda *args: "c" * 64
    )
    settings = load_extraction_settings({})
    ontology = SimpleNamespace(version="1.0", checksum="b" * 64)
    before = builds._document_context(
        document_id, "a" * 64, settings=settings, ontology=ontology
    )
    monkeypatch.setattr(identity_module, "ADAPTER_CONTRACT_VERSION", "adapter-next")
    after = builds._document_context(
        document_id, "a" * 64, settings=settings, ontology=ontology
    )
    assert builds.derive_document_build_key(
        before.identity
    ) != builds.derive_document_build_key(after.identity)


def test_evaluator_accepts_current_contract_and_rejects_legacy_identity():
    from apps.knowledge_graph.evals import run_kg_eval
    from apps.knowledge_graph.evals.fixture_manifest_types import (
        ResolvedFixtureManifest,
    )
    from apps.knowledge_graph.graph.filtering import (
        FilterPolicy,
        filter_policy_checksum,
    )
    from apps.knowledge_graph.models.inputs import collection_manifest_source_hash
    from apps.knowledge_graph.resolution.collection import (
        COLLECTION_RESOLVER_VERSION,
        CollectionResolutionConfig,
        resolution_config_checksum,
    )
    from lib.knowledge_graph.extraction_identity import extraction_identity

    settings = load_extraction_settings({})
    request = UUID("22222222-2222-4222-8222-222222222222")
    manifest = ResolvedFixtureManifest(
        fixture_id="contract-test",
        fixture_checksum="a" * 64,
        embedding=None,
        authorized_scope=((17, request),),
        collections={},
        documents={},
        chunks={},
        canonical_identity_assertions=(),
        inaccessible_neighbor_assertions=(),
        manifest_checksum="b" * 64,
    )
    policy = FilterPolicy()
    resolution = CollectionResolutionConfig()
    artifact = {
        "collection_scope_id": 17,
        "id": 1,
        "rebuild_request_id": request,
        "scope_type": "collection",
        "scope_id": "17",
        "status": "superseded",
        "evaluation_only": True,
        "ontology_version": "1.0",
        "ontology_checksum": "c" * 64,
        "extractor_version": extraction_identity(settings),
        "resolver_version": COLLECTION_RESOLVER_VERSION,
        "resolution_config_checksum": resolution_config_checksum(resolution),
        "filter_policy_version": policy.version,
        "filter_policy_checksum": filter_policy_checksum(policy),
        "source_hash": collection_manifest_source_hash(()),
        "embedding_model_signature": "fixture",
    }
    arguments = dict(
        manifest=manifest,
        selected_artifacts=(artifact,),
        manifest_rows=(),
        ontology=SimpleNamespace(version="1.0", checksum="c" * 64),
        extraction_settings=settings,
        filter_policy=policy,
        resolution_config=resolution,
        canonical_assertions=(),
    )
    assert (
        run_kg_eval._build_production_extraction_context(**arguments)[
            "extraction_settings"
        ]
        is settings
    )
    artifact["extractor_version"] = (
        f"{settings.provider}:{settings.model_id}@{settings.model_revision}"
    )
    with pytest.raises(run_kg_eval.ComparisonAborted, match="production provenance"):
        run_kg_eval._build_production_extraction_context(**arguments)
