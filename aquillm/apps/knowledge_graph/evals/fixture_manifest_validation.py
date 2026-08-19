"""Strict parser for the provider-neutral Task20 fixture manifest."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
from types import MappingProxyType
from uuid import UUID

from .fixture_manifest import (
    FixtureChunkBinding,
    FixtureCollectionBinding,
    FixtureDocumentBinding,
    FixtureEmbeddingBinding,
    assemble_fixture_document,
    embedding_endpoint_signature,
    fixture_manifest_checksum,
)
from .fixture_manifest_assertions import validate_fixture_assertions
from .fixture_manifest_types import FixtureValidationError, ResolvedFixtureManifest
from .fixture_manifest_validation_helpers import (
    exact_commit,
    exact_integer,
    exact_map,
    exact_sha,
    exact_text,
    exact_uuid,
    is_safe_huggingface_repo_id,
    logical_topology,
    symbol_map,
    validate_scope,
)

_FIXTURE_ID = "kg-task20-synthetic-v1"
_HIDDEN = "collection-security-private"


def validate_fixture_manifest_payload(
    manifest: object,
    *,
    extraction_cases: Sequence[Mapping[str, object]],
    retrieval_cases: Sequence[Mapping[str, object]],
    collection_requests: tuple[tuple[int, UUID], ...],
    expected_fixture_checksum: str,
) -> ResolvedFixtureManifest:
    top = exact_map(
        manifest,
        {
            "schema_version",
            "fixture_id",
            "fixture_checksum",
            "embedding",
            "authorized_scope",
            "collections",
            "documents",
            "chunks",
            "canonical_identity_assertions",
            "inaccessible_neighbor_assertions",
        },
        "fixture manifest",
    )
    if type(top["schema_version"]) is not int or top["schema_version"] != 1:
        raise FixtureValidationError("schema_version must be exact integer 1")
    if type(top["fixture_id"]) is not str or top["fixture_id"] != _FIXTURE_ID:
        raise FixtureValidationError("fixture manifest must be exact Task20 schema v1")
    logical_checksum = exact_sha(top["fixture_checksum"], "fixture_checksum")
    if logical_checksum != exact_sha(
        expected_fixture_checksum, "expected fixture checksum"
    ):
        raise FixtureValidationError("manifest differs from current fixture checksum")
    validate_scope(top, collection_requests)
    embedding_row = exact_map(
        top["embedding"],
        {"model", "checkpoint", "dimensions", "input_type", "endpoint_signature"},
        "embedding",
    )
    embedding_model = exact_text(embedding_row["model"], "embedding.model")
    if not is_safe_huggingface_repo_id(embedding_model):
        raise FixtureValidationError(
            "embedding.model must be a Hugging Face repository ID"
        )
    embedding = FixtureEmbeddingBinding(
        embedding_model,
        exact_commit(embedding_row["checkpoint"], "embedding.checkpoint"),
        exact_integer(embedding_row["dimensions"], "embedding.dimensions"),
        exact_text(embedding_row["input_type"], "embedding.input_type"),
        exact_sha(embedding_row["endpoint_signature"], "embedding.endpoint_signature"),
    )
    if embedding.dimensions != 1024 or embedding.input_type != "search_document":
        raise FixtureValidationError("embedding must be 1024-d search_document")
    if embedding.endpoint_signature != embedding_endpoint_signature(
        model=embedding.model,
        checkpoint=embedding.checkpoint,
        dimensions=embedding.dimensions,
        input_type=embedding.input_type,
    ):
        raise FixtureValidationError("embedding endpoint signature is inconsistent")
    logical = logical_topology(extraction_cases, retrieval_cases)
    (
        logical_collections,
        logical_documents,
        logical_chunks,
        expected_canonical,
        expected_hidden,
    ) = logical

    raw_collections = symbol_map(top["collections"], "collections")
    if set(raw_collections) != logical_collections:
        raise FixtureValidationError("collection symbols differ from current fixtures")
    collections: dict[str, FixtureCollectionBinding] = {}
    scope = set(collection_requests)
    for symbol, raw in raw_collections.items():
        row = exact_map(
            raw,
            {"collection_id", "rebuild_request_id", "authorized"},
            f"collection {symbol}",
        )
        authorized = row["authorized"]
        if type(authorized) is not bool:
            raise FixtureValidationError(
                f"collection {symbol} authorized must be boolean"
            )
        request = (
            None
            if row["rebuild_request_id"] is None
            else exact_uuid(
                row["rebuild_request_id"], f"collection {symbol}.rebuild_request_id"
            )
        )
        binding = FixtureCollectionBinding(
            symbol,
            exact_integer(row["collection_id"], f"collection {symbol}.collection_id"),
            request,
            authorized,
        )
        if symbol == _HIDDEN and (authorized or request is not None):
            raise FixtureValidationError(
                "hidden security collection must have no capability request"
            )
        if authorized != (
            request is not None and (binding.collection_id, request) in scope
        ):
            raise FixtureValidationError(
                "authorized iff exact request is in authorized_scope"
            )
        collections[symbol] = binding
    hidden = collections.get(_HIDDEN)
    if hidden is None or hidden.authorized or hidden.rebuild_request_id is not None:
        raise FixtureValidationError(
            "hidden security collection must have no capability request"
        )
    if hidden.collection_id in {row[0] for row in collection_requests}:
        raise FixtureValidationError(
            "hidden security collection must be outside capability"
        )
    if any(not row.authorized for key, row in collections.items() if key != _HIDDEN):
        raise FixtureValidationError(
            "all non-security fixture collections must be authorized"
        )
    if {
        (row.collection_id, row.rebuild_request_id)
        for row in collections.values()
        if row.authorized
    } != scope:
        raise FixtureValidationError(
            "authorized collection bindings differ from CLI scope"
        )
    if (
        not 2
        <= len({row.collection_id for row in collections.values() if row.authorized})
        <= 4
    ):
        raise FixtureValidationError(
            "fixture requires 2-4 authorized physical collections"
        )

    raw_documents = symbol_map(top["documents"], "documents")
    if set(raw_documents) != set(logical_documents):
        raise FixtureValidationError("document symbols differ from current fixtures")
    documents: dict[str, FixtureDocumentBinding] = {}
    assemblies: dict[str, tuple[str, tuple[tuple[int, int], ...]]] = {}
    for symbol, raw in raw_documents.items():
        row = exact_map(
            raw,
            {"document_id", "collection_symbol", "full_text_sha256"},
            f"document {symbol}",
        )
        collection_symbol, logical_rows = logical_documents[symbol]
        full_text, spans = assemble_fixture_document(
            tuple(text for _, text in logical_rows)
        )
        assemblies[symbol] = (full_text, spans)
        if (
            row["collection_symbol"] != collection_symbol
            or row["full_text_sha256"] != sha256(full_text.encode()).hexdigest()
        ):
            raise FixtureValidationError(
                f"document {symbol} differs from current fixture text/topology"
            )
        collection = collections[collection_symbol]
        documents[symbol] = FixtureDocumentBinding(
            symbol,
            exact_uuid(row["document_id"], f"document {symbol}.document_id"),
            collection_symbol,
            collection.collection_id,
            exact_sha(row["full_text_sha256"], f"document {symbol}.full_text_sha256"),
        )
    if len({row.document_id for row in documents.values()}) != len(documents):
        raise FixtureValidationError("document UUIDs must be unique")

    raw_chunks = symbol_map(top["chunks"], "chunks")
    if set(raw_chunks) != set(logical_chunks):
        raise FixtureValidationError("chunk symbols differ from current fixtures")
    chunks: dict[str, FixtureChunkBinding] = {}
    for symbol, raw in raw_chunks.items():
        row = exact_map(
            raw,
            {
                "chunk_id",
                "document_symbol",
                "chunk_number",
                "start",
                "end",
                "content_sha256",
                "embedding_sha256",
            },
            f"chunk {symbol}",
        )
        document_symbol, text, number = logical_chunks[symbol]
        start, end = assemblies[document_symbol][1][number]
        chunk_number = exact_integer(
            row["chunk_number"], f"chunk {symbol}.chunk_number", zero=True
        )
        observed_start = exact_integer(row["start"], f"chunk {symbol}.start", zero=True)
        observed_end = exact_integer(row["end"], f"chunk {symbol}.end")
        if (
            row["document_symbol"] != document_symbol
            or chunk_number != number
            or observed_start != start
            or observed_end != end
            or row["content_sha256"] != sha256(text.encode()).hexdigest()
            or assemblies[document_symbol][0][start:end] != text
        ):
            raise FixtureValidationError(
                f"chunk {symbol} differs from current fixture text/span topology"
            )
        document = documents[document_symbol]
        chunks[symbol] = FixtureChunkBinding(
            symbol,
            exact_integer(row["chunk_id"], f"chunk {symbol}.chunk_id"),
            document_symbol,
            document.collection_symbol,
            document.collection_id,
            number,
            start,
            end,
            exact_sha(row["content_sha256"], f"chunk {symbol}.content_sha256"),
            exact_sha(row["embedding_sha256"], f"chunk {symbol}.embedding_sha256"),
        )
    if len({row.chunk_id for row in chunks.values()}) != len(chunks):
        raise FixtureValidationError("chunk IDs must be unique")

    canonical, inaccessible = validate_fixture_assertions(
        canonical_rows=top["canonical_identity_assertions"],
        inaccessible_rows=top["inaccessible_neighbor_assertions"],
        expected_canonical=expected_canonical,
        expected_hidden=expected_hidden,
        collections=collections,
        chunks=chunks,
    )

    return ResolvedFixtureManifest(
        _FIXTURE_ID,
        logical_checksum,
        embedding,
        collection_requests,
        MappingProxyType(dict(sorted(collections.items()))),
        MappingProxyType(dict(sorted(documents.items()))),
        MappingProxyType(dict(sorted(chunks.items()))),
        canonical,
        inaccessible,
        fixture_manifest_checksum(top),
    )
