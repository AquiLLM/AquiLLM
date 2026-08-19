"""Immutable Task20 synthetic-fixture manifest contract.

This module is deliberately provider-, ORM-, and ML-runtime-free so both the
evaluation runner and the Task21 synthetic seeder use the same byte contract.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Mapping, Sequence
from hashlib import sha256
from math import isfinite
from pathlib import Path
from uuid import UUID

from .fixture_manifest_types import (
    FixtureCanonicalIdentityAssertion,
    FixtureChunkBinding,
    FixtureCollectionBinding,
    FixtureDocumentBinding,
    FixtureEmbeddingBinding,
    FixtureInaccessibleNeighborAssertion,
    FixtureValidationError,
    ResolvedFixtureManifest,
)
from .fixture_manifest_validation_helpers import is_safe_huggingface_repo_id

_SHA256_LENGTH = 64
_FIXTURE_ID = "kg-task20-synthetic-v1"
_HIDDEN_COLLECTION = "collection-security-private"
_HERE = Path(__file__).resolve().parent
_DEFAULT_EXTRACTION_CASES_PATH = _HERE / "extraction_cases.yaml"
_DEFAULT_RETRIEVAL_CASES_PATH = _HERE / "retrieval_cases.yaml"


def assemble_fixture_document(
    chunk_texts: tuple[str, ...],
) -> tuple[str, tuple[tuple[int, int], ...]]:
    """Assemble chunks by maximal suffix/prefix overlap, else two newlines."""

    if type(chunk_texts) is not tuple or not chunk_texts:
        raise FixtureValidationError("fixture document chunks must be a nonempty tuple")
    if any(type(text) is not str or not text for text in chunk_texts):
        raise FixtureValidationError(
            "fixture chunk text must be a nonempty exact string"
        )
    full_text = chunk_texts[0]
    spans: list[tuple[int, int]] = [(0, len(full_text))]
    for text in chunk_texts[1:]:
        maximum = min(len(full_text), len(text))
        overlap = max(
            (
                size
                for size in range(1, maximum + 1)
                if full_text[-size:] == text[:size]
            ),
            default=0,
        )
        if overlap == len(text):
            raise FixtureValidationError(
                "fixture chunk is wholly contained by its predecessor"
            )
        if overlap:
            start = len(full_text) - overlap
            full_text += text[overlap:]
        else:
            start = len(full_text) + 2
            full_text += "\n\n" + text
        end = start + len(text)
        if full_text[start:end] != text:
            raise FixtureValidationError("fixture overlap assembly is ambiguous")
        spans.append((start, end))
    return full_text, tuple(spans)


def canonical_embedding_sha256(vector: tuple[object, ...]) -> str:
    """Hash exact pgvector semantics as 1024 big-endian IEEE-754 binary32s."""

    if type(vector) is not tuple or len(vector) != 1024:
        raise FixtureValidationError("fixture embedding must be an exact 1024-d tuple")
    encoded = bytearray()
    for value in vector:
        if type(value) not in (int, float) or not isfinite(float(value)):
            raise FixtureValidationError("fixture embedding values must be finite")
        try:
            packed = struct.pack(">f", float(value))
        except (OverflowError, struct.error) as error:
            raise FixtureValidationError(
                "fixture embedding exceeds binary32"
            ) from error
        if not isfinite(struct.unpack(">f", packed)[0]):
            raise FixtureValidationError("fixture embedding exceeds binary32")
        encoded.extend(packed)
    return sha256(encoded).hexdigest()


def embedding_endpoint_signature(
    *,
    model: str,
    checkpoint: str,
    dimensions: int,
    input_type: str,
) -> str:
    """Sign the exact non-secret endpoint contract used for fixture vectors."""

    if (
        type(model) is not str
        or not model
        or type(checkpoint) is not str
        or not checkpoint
        or type(dimensions) is not int
        or dimensions < 1
        or type(input_type) is not str
        or not input_type
    ):
        raise FixtureValidationError("embedding endpoint contract is malformed")
    encoded = json.dumps(
        {
            "checkpoint": checkpoint,
            "dimensions": dimensions,
            "input_type": input_type,
            "model": model,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return sha256(encoded).hexdigest()


def _canonical_json_value(value: object) -> object:
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is list:
        return [_canonical_json_value(item) for item in value]
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise FixtureValidationError("fixture manifest keys must be exact strings")
        return {key: _canonical_json_value(value[key]) for key in sorted(value)}
    raise FixtureValidationError(
        f"fixture manifest contains unsupported {type(value).__name__}"
    )


def fixture_manifest_checksum(manifest: Mapping[str, object]) -> str:
    """Hash canonical JSON; the resolved manifest never contains this digest."""

    if type(manifest) is not dict:
        raise FixtureValidationError("fixture manifest must be an exact JSON object")
    encoded = json.dumps(
        _canonical_json_value(manifest),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def fixture_checksum(
    extraction_path: Path = _DEFAULT_EXTRACTION_CASES_PATH,
    retrieval_path: Path = _DEFAULT_RETRIEVAL_CASES_PATH,
) -> str:
    """Checksum the exact named fixture bytes in deterministic order."""

    digest = sha256()
    for index, (name, path) in enumerate(
        (
            ("extraction_cases.yaml", extraction_path),
            ("retrieval_cases.yaml", retrieval_path),
        )
    ):
        try:
            contents = path.read_bytes()
        except OSError as error:
            raise FixtureValidationError(
                f"could not read fixture {path}: {error}"
            ) from error
        if index:
            digest.update(b"\x00")
        digest.update(name.encode("ascii"))
        digest.update(b"\x00")
        digest.update(contents)
    return digest.hexdigest()


def _object_without_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise FixtureValidationError(f"fixture manifest duplicates key {key!r}")
        result[key] = value
    return result


def load_fixture_manifest(path: Path) -> dict[str, object]:
    """Load strict UTF-8 JSON with duplicate-key rejection."""

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_object_without_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                FixtureValidationError(
                    f"fixture manifest contains invalid constant {value!r}"
                )
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FixtureValidationError(
            f"could not load fixture manifest {path}: {error}"
        ) from error
    if type(payload) is not dict:
        raise FixtureValidationError("fixture manifest must be an exact JSON object")
    return payload


def validate_fixture_manifest(
    manifest: object,
    *,
    extraction_cases: Sequence[Mapping[str, object]],
    retrieval_cases: Sequence[Mapping[str, object]],
    collection_requests: tuple[tuple[int, UUID], ...],
    expected_fixture_checksum: str,
) -> ResolvedFixtureManifest:
    """Validate through the compact provider-neutral implementation helper."""

    from .fixture_manifest_validation import validate_fixture_manifest_payload

    return validate_fixture_manifest_payload(
        manifest,
        extraction_cases=extraction_cases,
        retrieval_cases=retrieval_cases,
        collection_requests=collection_requests,
        expected_fixture_checksum=expected_fixture_checksum,
    )


__all__ = (
    "FixtureCanonicalIdentityAssertion",
    "FixtureChunkBinding",
    "FixtureCollectionBinding",
    "FixtureDocumentBinding",
    "FixtureEmbeddingBinding",
    "FixtureInaccessibleNeighborAssertion",
    "FixtureValidationError",
    "ResolvedFixtureManifest",
    "assemble_fixture_document",
    "canonical_embedding_sha256",
    "embedding_endpoint_signature",
    "fixture_checksum",
    "fixture_manifest_checksum",
    "is_safe_huggingface_repo_id",
    "load_fixture_manifest",
    "validate_fixture_manifest",
)
