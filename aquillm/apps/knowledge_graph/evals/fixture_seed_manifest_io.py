"""Embedding and canonical manifest I/O for the synthetic fixture."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from uuid import UUID

from .fixture_manifest import (
    FixtureValidationError,
    ResolvedFixtureManifest,
    canonical_embedding_sha256,
    embedding_endpoint_signature,
    fixture_manifest_checksum,
    load_fixture_manifest,
    validate_fixture_manifest,
)
from .fixture_seed_contract import (
    DIMENSIONS,
    INPUT_TYPE,
    EmbeddingIdentity,
    FixtureSeedError,
    FixtureSeedResult,
    LogicalFixture,
)


def embedding_vectors(
    logical: LogicalFixture,
    model: str,
    checkpoint: str,
    *,
    signature_loader,
    embedding_loader,
):
    try:
        model_signature = signature_loader()
    except Exception as error:
        raise FixtureSeedError("strict local embedding identity failed") from error
    prefix = f"local-openai:{model}@{checkpoint}:"
    if (
        type(model_signature) is not str
        or not model_signature.startswith(prefix)
        or ":dims=1024:" not in model_signature
        or len(model_signature) > 1_024
    ):
        raise FixtureSeedError("strict local embedding signature is inconsistent")
    texts = [logical.chunks[symbol][1] for symbol in sorted(logical.chunks)]
    try:
        indexed, actual_signature = embedding_loader(
            texts,
            expected_model_signature=model_signature,
        )
    except Exception as error:
        raise FixtureSeedError("strict local embedding request failed") from error
    if actual_signature != model_signature or len(indexed) != len(texts):
        raise FixtureSeedError("strict local embedding response is inconsistent")
    by_index: dict[int, tuple[float, ...]] = {}
    try:
        for index, vector in indexed:
            if (
                type(index) is not int
                or index in by_index
                or not 0 <= index < len(texts)
            ):
                raise FixtureValidationError("embedding index is invalid")
            canonical = tuple(vector)
            canonical_embedding_sha256(canonical)
            by_index[index] = canonical
    except (TypeError, ValueError, FixtureValidationError) as error:
        raise FixtureSeedError("strict local embedding vector is invalid") from error
    if set(by_index) != set(range(len(texts))):
        raise FixtureSeedError("strict local embedding indices are incomplete")
    identity = EmbeddingIdentity(
        model,
        checkpoint,
        model_signature,
        embedding_endpoint_signature(
            model=model,
            checkpoint=checkpoint,
            dimensions=DIMENSIONS,
            input_type=INPUT_TYPE,
        ),
    )
    return identity, {
        symbol: by_index[index] for index, symbol in enumerate(sorted(logical.chunks))
    }


def canonical_manifest_bytes(payload: dict[str, object]) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def manifest_scope(payload: dict[str, object]) -> tuple[tuple[int, UUID], ...]:
    raw_scope = payload.get("authorized_scope")
    if type(raw_scope) is not list:
        raise FixtureSeedError("fixture manifest scope is invalid")
    result: list[tuple[int, UUID]] = []
    try:
        for row in raw_scope:
            if type(row) is not dict or set(row) != {
                "collection_id",
                "rebuild_request_id",
            }:
                raise ValueError
            collection_id = row["collection_id"]
            request_text = row["rebuild_request_id"]
            if (
                type(collection_id) is not int
                or collection_id < 1
                or type(request_text) is not str
            ):
                raise ValueError
            request_id = UUID(request_text)
            if str(request_id) != request_text:
                raise ValueError
            result.append((collection_id, request_id))
    except (AttributeError, TypeError, ValueError) as error:
        raise FixtureSeedError("fixture manifest scope is invalid") from error
    return tuple(result)


def validate_payload(
    payload: dict[str, object], logical: LogicalFixture
) -> ResolvedFixtureManifest:
    try:
        return validate_fixture_manifest(
            payload,
            extraction_cases=logical.extraction_cases,
            retrieval_cases=logical.retrieval_cases,
            collection_requests=manifest_scope(payload),
            expected_fixture_checksum=logical.checksum,
        )
    except FixtureValidationError as error:
        raise FixtureSeedError("fixture manifest validation failed") from error


def atomic_publish_manifest(
    path: Path,
    payload: dict[str, object],
    logical: LogicalFixture,
) -> ResolvedFixtureManifest:
    encoded = canonical_manifest_bytes(payload)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    published = False
    try:
        os.chmod(temporary, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        loaded = load_fixture_manifest(temporary)
        resolved = validate_payload(loaded, logical)
        if loaded != payload or resolved.manifest_checksum != fixture_manifest_checksum(
            payload
        ):
            raise FixtureSeedError("fixture manifest round-trip failed")
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FixtureSeedError("fixture manifest already exists") from error
        published = True
        return resolved
    except (OSError, UnicodeError, FixtureValidationError) as error:
        raise FixtureSeedError("manifest publication failed") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            if not published:
                raise FixtureSeedError("manifest publication cleanup failed")


def seed_result(resolved: ResolvedFixtureManifest, path: Path) -> FixtureSeedResult:
    return FixtureSeedResult(
        resolved.fixture_checksum,
        resolved.manifest_checksum,
        path,
        resolved.authorized_scope,
        len({binding.collection_id for binding in resolved.collections.values()}),
        len(resolved.documents),
        len(resolved.chunks),
    )
