"""Validate present metadata even when its projection is not ready to serve."""

from .ready_scope import _checksum


def validate_present_metadata(*, projection, membership, artifact):
    projection.clean()
    membership.clean()
    for name in (
        "source_hash",
        "build_key",
        "ontology_checksum",
        "resolution_config_checksum",
        "filter_policy_checksum",
        "assembly_config_checksum",
    ):
        _checksum(getattr(artifact, name), name)
    for name in (
        "ontology_version",
        "extractor_version",
        "resolver_version",
        "filter_policy_version",
        "embedding_model_signature",
        "assembly_version",
    ):
        token = getattr(artifact, name)
        if (
            type(token) is not str
            or not token
            or token != token.strip()
            or len(token) > 512
            or any(c in token for c in "\x00\r\n")
        ):
            raise ValueError("malformed present artifact metadata")
    if projection.representation_digest != artifact.representation_digest:
        raise ValueError("projection representation differs from its artifact")
