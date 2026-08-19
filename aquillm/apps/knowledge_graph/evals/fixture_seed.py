"""Public service seam for deterministic synthetic KG fixture lifecycle."""

from __future__ import annotations

import secrets

from django.db import transaction

from apps.knowledge_graph.graph.assembly import lock_collection_graph_advisory_scope
from aquillm.utils import get_strict_index_embeddings, strict_index_embedding_signature

from .fixture_manifest import (
    FixtureValidationError,
    fixture_manifest_checksum,
    load_fixture_manifest,
)
from .fixture_seed_cases import logical_fixture
from .fixture_seed_contract import (
    FIXTURE_ID,
    SHA_PATTERN,
    FixtureSeedError,
    FixtureSeedResult,
    manifest_path,
    require_safe_environment,
)
from .fixture_seed_graph_requests import physical_labels
from .fixture_seed_manifest_io import (
    atomic_publish_manifest,
    canonical_manifest_bytes,
    embedding_vectors,
    seed_result,
    validate_payload,
)
from .fixture_seed_persistence import assert_no_identity_collisions, persist_fixture
from .fixture_seed_query import bounded_rows
from .fixture_seed_topology import validate_owned_topology

# Deliberate private test seam for atomic-publication fault injection.
_atomic_publish_manifest = atomic_publish_manifest


def _load_existing(path, logical):
    try:
        payload = load_fixture_manifest(path)
    except FixtureValidationError as error:
        raise FixtureSeedError(
            "fixture manifest already exists but is not command-owned"
        ) from error
    if path.read_bytes() != canonical_manifest_bytes(payload):
        raise FixtureSeedError("fixture manifest already exists but is not canonical")
    resolved = validate_payload(payload, logical)
    validate_owned_topology(resolved, logical)
    return resolved


def seed_fixture(fixture_manifest: str) -> FixtureSeedResult:
    model, checkpoint = require_safe_environment()
    path = manifest_path(fixture_manifest, must_exist=False)
    logical = logical_fixture()
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise FixtureSeedError("fixture manifest already exists")
        return seed_result(_load_existing(path, logical), path)
    assert_no_identity_collisions(logical)
    embedding, vectors = embedding_vectors(
        logical,
        model,
        checkpoint,
        signature_loader=strict_index_embedding_signature,
        embedding_loader=get_strict_index_embeddings,
    )
    resolved = persist_fixture(
        path,
        logical,
        embedding,
        vectors,
        publish=_atomic_publish_manifest,
    )
    return seed_result(resolved, path)


def _load_cleanup_manifest(path, expected_checksum, logical):
    try:
        payload = load_fixture_manifest(path)
    except FixtureValidationError as error:
        raise FixtureSeedError("fixture manifest validation failed") from error
    actual_checksum = fixture_manifest_checksum(payload)
    if not secrets.compare_digest(actual_checksum, expected_checksum):
        raise FixtureSeedError(
            "fixture manifest checksum does not match expected checksum"
        )
    if path.read_bytes() != canonical_manifest_bytes(payload):
        raise FixtureSeedError("fixture manifest is not canonical")
    return validate_payload(payload, logical)


def cleanup_fixture(
    fixture_manifest: str,
    *,
    expected_manifest_checksum: str,
) -> FixtureSeedResult:
    require_safe_environment()
    if (
        type(expected_manifest_checksum) is not str
        or SHA_PATTERN.fullmatch(expected_manifest_checksum) is None
    ):
        raise FixtureSeedError("expected manifest checksum must be lowercase SHA-256")
    path = manifest_path(fixture_manifest, must_exist=True)
    logical = logical_fixture()
    resolved = _load_cleanup_manifest(path, expected_manifest_checksum, logical)
    expected_ids = tuple(sorted(physical_labels(resolved)))
    deleted = 0
    with transaction.atomic():
        for collection_id in expected_ids:
            lock_collection_graph_advisory_scope(collection_id)
        from apps.collections.models import Collection

        locked = bounded_rows(
            Collection.objects.select_for_update().filter(pk__in=expected_ids),
            len(expected_ids),
        )
        collections, users = validate_owned_topology(
            resolved,
            logical,
            allow_absent=True,
            locked_collections=locked,
        )
        if collections:
            target_ids = tuple(sorted(collection.pk for collection in collections))
            if target_ids != expected_ids:
                raise FixtureSeedError("fixture database topology is not exact")
            Collection.objects.filter(pk__in=target_ids).delete()
            deleted = len(target_ids)
            for user in users:
                user.delete()
    result = seed_result(resolved, path)
    return FixtureSeedResult(
        result.fixture_checksum,
        result.manifest_checksum,
        result.manifest_path,
        result.authorized_scope,
        deleted,
        result.document_count if deleted else 0,
        result.chunk_count if deleted else 0,
    )


__all__ = (
    "FIXTURE_ID",
    "FixtureSeedError",
    "FixtureSeedResult",
    "cleanup_fixture",
    "seed_fixture",
)
