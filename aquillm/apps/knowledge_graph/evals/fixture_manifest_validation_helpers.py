"""Primitive and logical-topology checks for fixture manifest validation."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from uuid import UUID

from .fixture_manifest_types import FixtureValidationError

_SHA = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_HUGGINGFACE_REPO_ID = re.compile(
    r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,126}[A-Za-z0-9])?/"
    r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,126}[A-Za-z0-9])?"
)


def is_safe_huggingface_repo_id(value: object) -> bool:
    """Return whether value is a bounded immutable-Hub-compatible repo ID."""

    return (
        type(value) is str
        and len(value) <= 256
        and _HUGGINGFACE_REPO_ID.fullmatch(value) is not None
    )


def exact_map(value: object, keys: set[str], context: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != keys:
        raise FixtureValidationError(f"{context} has non-exact fields")
    return value


def symbol_map(value: object, context: str) -> dict[str, object]:
    if (
        type(value) is not dict
        or not value
        or any(type(key) is not str or not key for key in value)
    ):
        raise FixtureValidationError(f"{context} must be a nonempty symbol object")
    if tuple(value) != tuple(sorted(value)):
        raise FixtureValidationError(f"{context} symbols must be sorted and unique")
    return value


def exact_sequence(value: object, context: str) -> list[object]:
    if type(value) is not list or not value:
        raise FixtureValidationError(f"{context} must be a nonempty exact list")
    return value


def exact_text(value: object, context: str) -> str:
    if type(value) is not str or not value or value != value.strip() or "\x00" in value:
        raise FixtureValidationError(f"{context} must be a nonempty exact string")
    return value


def exact_integer(value: object, context: str, *, zero: bool = False) -> int:
    minimum = 0 if zero else 1
    if type(value) is not int or value < minimum:
        raise FixtureValidationError(f"{context} must be an exact integer >= {minimum}")
    return value


def exact_sha(value: object, context: str) -> str:
    if type(value) is not str or _SHA.fullmatch(value) is None:
        raise FixtureValidationError(f"{context} must be a lowercase SHA-256")
    return value


def exact_commit(value: object, context: str) -> str:
    if type(value) is not str or _COMMIT.fullmatch(value) is None:
        raise FixtureValidationError(f"{context} must be a lowercase 40-hex commit")
    return value


def exact_uuid(value: object, context: str) -> UUID:
    if type(value) is not str:
        raise FixtureValidationError(f"{context} must be a canonical UUID")
    try:
        parsed = UUID(value)
    except ValueError as error:
        raise FixtureValidationError(f"{context} must be a canonical UUID") from error
    if str(parsed) != value:
        raise FixtureValidationError(f"{context} must be a canonical UUID")
    return parsed


def logical_topology(
    extraction_cases: Sequence[Mapping[str, object]],
    retrieval_cases: Sequence[Mapping[str, object]],
):
    collections: set[str] = set()
    documents: dict[str, tuple[str, tuple[tuple[str, str], ...]]] = {}
    chunks: dict[str, tuple[str, str, int]] = {}
    canonical: set[tuple[str, str]] = set()
    inaccessible: set[tuple[str, str]] = set()
    for case in (*extraction_cases, *retrieval_cases):
        accessible = set(case.get("accessible_collection_ids", ()))
        collections.update(accessible)
        accessible_chunks: list[str] = []
        hidden_chunks: list[str] = []
        for document in case["documents"]:  # type: ignore[index]
            symbol = document["doc_id"]
            collection = document["collection_id"]
            collections.add(collection)
            rows = tuple(
                (chunk["chunk_id"], chunk["text"]) for chunk in document["chunks"]
            )
            if documents.setdefault(symbol, (collection, rows)) != (collection, rows):
                raise FixtureValidationError(
                    f"logical document {symbol!r} is ambiguous"
                )
            for number, (chunk_symbol, text) in enumerate(rows):
                if chunks.setdefault(chunk_symbol, (symbol, text, number)) != (
                    symbol,
                    text,
                    number,
                ):
                    raise FixtureValidationError(
                        f"logical chunk {chunk_symbol!r} is ambiguous"
                    )
                (
                    accessible_chunks if collection in accessible else hidden_chunks
                ).append(chunk_symbol)
        for link in case.get("canonical_identity_links", ()):  # type: ignore[union-attr]
            canonical.add((link["source_chunk_id"], link["target_chunk_id"]))
        if "inaccessible_neighbor" in case.get("quality_tags", ()):  # type: ignore[union-attr]
            inaccessible.update(
                (source, target)
                for source in accessible_chunks
                for target in hidden_chunks
            )
    return collections, documents, chunks, canonical, inaccessible


def validate_scope(
    manifest: dict[str, object],
    collection_requests: tuple[tuple[int, UUID], ...],
) -> None:
    if (
        type(collection_requests) is not tuple
        or not 1 <= len(collection_requests) <= 4
        or any(
            type(row) is not tuple
            or len(row) != 2
            or type(row[0]) is not int
            or row[0] < 1
            or type(row[1]) is not UUID
            for row in collection_requests
        )
        or collection_requests != tuple(sorted(collection_requests))
        or len(set(collection_requests)) != len(collection_requests)
        or len({row[0] for row in collection_requests}) != len(collection_requests)
        or len({row[1] for row in collection_requests}) != len(collection_requests)
    ):
        raise FixtureValidationError(
            "CLI collection/request scope must be sorted unique 1-4"
        )
    observed: list[tuple[int, UUID]] = []
    for index, raw in enumerate(
        exact_sequence(manifest["authorized_scope"], "authorized_scope")
    ):
        row = exact_map(raw, {"collection_id", "rebuild_request_id"}, f"scope[{index}]")
        observed.append(
            (
                exact_integer(row["collection_id"], f"scope[{index}].collection_id"),
                exact_uuid(row["rebuild_request_id"], f"scope[{index}].request"),
            )
        )
    if tuple(observed) != collection_requests:
        raise FixtureValidationError(
            "manifest scope differs from CLI collection/request pairs"
        )
