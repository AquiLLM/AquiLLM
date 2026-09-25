"""Related graph infrastructure regression scenarios."""

from apps.knowledge_graph.tests.test_projection_records import (
    _bundle,
    pytest,
    replace,
)


def test_document_provenance_allows_stage_specific_resolution_and_filtering() -> None:
    bundle = _bundle()
    document = replace(
        bundle.artifact_provenance[1],
        resolver_version="document-coreference-v1",
        resolution_config_checksum="e" * 64,
        filter_policy_version="pending-v1",
        filter_policy_checksum="f" * 64,
    )

    updated = replace(
        bundle,
        artifact_provenance=(bundle.artifact_provenance[0], document),
    )

    assert updated.artifact_provenance[1] == document



def test_document_provenance_allows_empty_embedding_signature() -> None:
    document = _bundle().artifact_provenance[1]
    assert document.embedding_model_signature == ""
    with pytest.raises(ValueError, match="document embedding"):
        replace(document, embedding_model_signature="embed-v1")
    with pytest.raises(ValueError, match="collection embedding"):
        replace(_bundle().artifact_provenance[0], embedding_model_signature="")
    subclass = type("_SignatureSubclass", (str,), {})("")
    with pytest.raises(TypeError, match="built-in str"):
        replace(document, embedding_model_signature=subclass)
