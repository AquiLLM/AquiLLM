"""Canonical cache identity for the complete frozen retrieval scope."""

import json
from hashlib import sha256


def retrieval_scope_key(
    *,
    authorization_digest: str,
    representation_digest: str,
    dense_generation_digest: str,
    embedding_space_key: str | None,
    graph_authority_digest: str,
    graph_policy_digest: str,
    query_digest: str,
    ranking_policy: str,
    vocabulary_digest: str,
) -> str:
    payload = locals()
    for name, value in payload.items():
        if name == "embedding_space_key" and value is None:
            continue
        if type(value) is not str or not value.strip():
            raise ValueError("invalid_retrieval_scope_identity")
    return sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
