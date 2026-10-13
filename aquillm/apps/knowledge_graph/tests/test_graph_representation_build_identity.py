"""Representation inputs enter build identity without changing resolver policy."""

import json
from dataclasses import asdict, replace
from hashlib import sha256

from apps.knowledge_graph.services import builds
from apps.knowledge_graph.tests.test_build_idempotency import (
    _collection_identity,
    _document_identity,
)


def test_absent_representation_keeps_exact_legacy_document_hash():
    identity = _document_identity()
    old = asdict(identity)
    old.pop("representation_digest", None)
    expected = sha256(
        json.dumps(
            {"namespace": "kg-document-build-v1", "identity": old},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
    ).hexdigest()
    assert builds.derive_document_build_key(identity) == expected


def test_representation_changes_key_without_fragmenting_resolver_compatibility():
    for identity, derive in (
        (_document_identity(), builds.derive_document_build_key),
        (_collection_identity(), builds.derive_collection_build_key),
    ):
        first = replace(identity, representation_digest="a" * 64)
        second = replace(identity, representation_digest="b" * 64)
        assert len({derive(identity), derive(first), derive(second)}) == 3
        assert (
            first.resolver_checksum
            == second.resolver_checksum
            == identity.resolver_checksum
        )
