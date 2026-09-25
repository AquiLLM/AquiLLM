"""Frozen YAML-to-logical-fixture assembly."""

from __future__ import annotations

from .fixture_manifest import fixture_checksum
from .fixture_seed_contract import (
    PHYSICAL_BINDINGS,
    FixtureSeedError,
    LogicalChunk,
    LogicalDocument,
    LogicalFixture,
    safe_token,
)
from .run_kg_eval import load_extraction_cases, load_retrieval_cases


def logical_fixture() -> LogicalFixture:
    extraction_cases = load_extraction_cases()
    retrieval_cases = load_retrieval_cases()
    records: dict[str, tuple[str, tuple[LogicalChunk, ...], str | None]] = {}
    chunks: dict[str, tuple[str, str, int]] = {}
    canonical: set[tuple[str, str]] = set()
    inaccessible: set[tuple[str, str]] = set()
    collections: set[str] = set()
    for case in (*extraction_cases, *retrieval_cases):
        accessible = set(case.get("accessible_collection_ids", ()))
        accessible_chunks: list[str] = []
        hidden_chunks: list[str] = []
        for raw_document in case["documents"]:
            symbol = raw_document["doc_id"]
            collection_symbol = raw_document["collection_id"]
            collections.add(collection_symbol)
            logical_chunks = tuple(
                LogicalChunk(chunk["chunk_id"], chunk["text"])
                for chunk in raw_document["chunks"]
            )
            title = raw_document.get("title")
            if title is not None and not safe_token(title):
                raise FixtureSeedError("synthetic fixture title is invalid")
            previous = records.get(symbol)
            if previous is None:
                records[symbol] = (collection_symbol, logical_chunks, title)
            elif previous[:2] != (collection_symbol, logical_chunks):
                raise FixtureSeedError(
                    "synthetic fixture document topology is ambiguous"
                )
            elif previous[2] not in (None, title) and title is not None:
                raise FixtureSeedError("synthetic fixture document title is ambiguous")
            elif previous[2] is None and title is not None:
                records[symbol] = (collection_symbol, logical_chunks, title)
            for number, chunk in enumerate(logical_chunks):
                row = (symbol, chunk.text, number)
                if chunks.setdefault(chunk.symbol, row) != row:
                    raise FixtureSeedError(
                        "synthetic fixture chunk topology is ambiguous"
                    )
                target = (
                    accessible_chunks
                    if collection_symbol in accessible
                    else hidden_chunks
                )
                target.append(chunk.symbol)
        canonical.update(
            (link["source_chunk_id"], link["target_chunk_id"])
            for link in case.get("canonical_identity_links", ())
        )
        if "inaccessible_neighbor" in case.get("quality_tags", ()):
            inaccessible.update(
                (source, target)
                for source in accessible_chunks
                for target in hidden_chunks
            )
    if collections != set(PHYSICAL_BINDINGS):
        raise FixtureSeedError("synthetic fixture collection topology has drifted")
    documents = {
        symbol: LogicalDocument(
            symbol,
            collection_symbol,
            title or f"Task20 synthetic evaluation {symbol}",
            logical_chunks,
        )
        for symbol, (collection_symbol, logical_chunks, title) in sorted(
            records.items()
        )
    }
    return LogicalFixture(
        extraction_cases,
        retrieval_cases,
        documents,
        dict(sorted(chunks.items())),
        tuple(sorted(canonical)),
        tuple(sorted(inaccessible)),
        fixture_checksum(),
    )
