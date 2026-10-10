"""Build/persistence adapters retain one captured vocabulary policy identity."""
from dataclasses import replace

from .vocabulary_capture import (
    KEY, VocabularyGraphUnavailable, capture_build_vocabulary, metadata, validate_artifact,
)
from .vocabulary_policy import bind_digest


def document_capsule(document_id, *, using="default"):
    from apps.knowledge_graph.extraction.pipeline import _get_concrete_document
    from .vocabulary_capture import enabled
    if not enabled():
        return None
    document = _get_concrete_document(document_id)
    capture = capture_build_vocabulary((document.collection_id,), using=using)
    return capture.get(document.collection_id)


def document_artifact_policy(artifact, *, using="default"):
    from .vocabulary_capture import enabled
    if not enabled() and KEY not in (getattr(artifact, "metadata", None) or {}):
        return None
    from apps.knowledge_graph.extraction.pipeline import _get_concrete_document
    document = _get_concrete_document(artifact.scope_id)
    capture = capture_build_vocabulary((document.collection_id,), using=using)
    return validate_artifact(artifact, capture, collection_id=document.collection_id)


def bind_collection_config(config, collection_id, artifacts=(), *, using="default"):
    capture = capture_build_vocabulary((collection_id,), using=using)
    capsule = capture.get(collection_id) if capture is not None else None
    policy = capsule.policy if capsule else None
    if config.vocabulary is not None and config.vocabulary != policy:
        raise VocabularyGraphUnavailable("collection vocabulary changed")
    for artifact in artifacts:
        validate_artifact(artifact, capture, collection_id=collection_id)
    return replace(config, vocabulary=policy), capsule


def collection_metadata(capsule, config):
    from .collection import resolution_config_checksum
    return metadata(capsule, resolution_config_checksum(replace(config, vocabulary=None)))


def validate_collection_result(artifact, config):
    from .vocabulary_capture import enabled
    if not enabled() and config.vocabulary is None and KEY not in (getattr(artifact, "metadata", None) or {}):
        return
    capture = capture_build_vocabulary((artifact.collection_scope_id,), using=artifact._state.db or "default")
    policy = validate_artifact(artifact, capture, collection_id=artifact.collection_scope_id)
    if policy != config.vocabulary:
        raise VocabularyGraphUnavailable("collection result vocabulary changed")


def capture_canonical_artifacts(artifacts, *, using):
    def value(row, key):
        return row[key] if type(row) is dict else getattr(row, key)
    capture = capture_build_vocabulary(tuple(value(a, "collection_scope_id") for a in artifacts), using=using)
    for artifact in artifacts:
        validate_artifact(artifact, capture, collection_id=value(artifact, "collection_scope_id"))
    return capture.policy if capture else None
