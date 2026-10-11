"""Contextual graph identity regression controls extracted from legacy suites."""
from types import SimpleNamespace

from test_build_idempotency import DOCUMENT_ID


def test_document_context_binds_effective_contextual_policy(monkeypatch):
    from test_coreference import _ontology

    from apps.knowledge_graph.extraction import pipeline
    from apps.knowledge_graph.services import builds

    ontology = _ontology()
    ontology.version = "1.0.0"

    monkeypatch.setattr(
        pipeline,
        "_get_concrete_document",
        lambda *_args, **_kwargs: SimpleNamespace(
            id=DOCUMENT_ID,
            collection_id=7,
            _meta=SimpleNamespace(label_lower="documents.pdfdocument"),
        ),
    )
    monkeypatch.setattr(pipeline, "_validate_source", lambda *_args: None)
    monkeypatch.setattr(pipeline, "_ordered_chunks", lambda *_args, **_kwargs: ())
    monkeypatch.setattr(
        builds, "ordered_chunk_signature", lambda *_args, **_kwargs: "b" * 64
    )
    monkeypatch.setattr(
        builds, "_ontology_activation_signature", lambda *_args: "c" * 64
    )
    monkeypatch.setenv("KG_IDENTITY_POLICY", "legacy")
    legacy = builds._document_context(DOCUMENT_ID, "a" * 64, ontology=ontology)
    monkeypatch.setenv("KG_IDENTITY_POLICY", "contextual_v2")
    contextual = builds._document_context(DOCUMENT_ID, "a" * 64, ontology=ontology)
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
