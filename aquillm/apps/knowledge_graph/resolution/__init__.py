"""Deterministic, provider-neutral document and collection resolution."""

CANONICAL_RESOLVER_VERSION = "canonical-resolution-v1"
COLLECTION_RESOLVER_VERSION = "collection-resolution-v1"
DOCUMENT_RESOLVER_VERSION = "document-coreference-v3-sparse-acronym"
DOCUMENT_CONTEXTUAL_RESOLVER_VERSION = "document-coreference-v4-contextual"
COLLECTION_CONTEXTUAL_RESOLVER_VERSION = "collection-resolution-v2-contextual"
CANONICAL_CONTEXTUAL_RESOLVER_VERSION = "canonical-resolution-v2-contextual"


def resolver_version_for_policy(tier: str, identity_policy: str) -> str:
    if type(identity_policy) is not str or identity_policy not in (
        "legacy",
        "contextual_v2",
    ):
        raise ValueError("identity_policy must be legacy or contextual_v2")
    versions = {
        "document": (DOCUMENT_RESOLVER_VERSION, DOCUMENT_CONTEXTUAL_RESOLVER_VERSION),
        "collection": (
            COLLECTION_RESOLVER_VERSION,
            COLLECTION_CONTEXTUAL_RESOLVER_VERSION,
        ),
        "canonical": (
            CANONICAL_RESOLVER_VERSION,
            CANONICAL_CONTEXTUAL_RESOLVER_VERSION,
        ),
    }
    return versions[tier][identity_policy == "contextual_v2"]


__all__ = [
    "CANONICAL_RESOLVER_VERSION",
    "COLLECTION_RESOLVER_VERSION",
    "DOCUMENT_RESOLVER_VERSION",
]
