from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from apps.knowledge_graph.evals import task21_hybrid_live_authority as authority


def test_live_fixture_rows_are_bound_to_manifest_hashes_and_metadata():
    vector = (0.0,) * 1024
    document = SimpleNamespace(
        id="doc-id",
        collection_id=7,
        full_text="document text",
        full_text_hash=hashlib.sha256(b"document text").hexdigest(),
    )
    chunk = SimpleNamespace(
        pk=11,
        doc_id="doc-id",
        chunk_number=0,
        content="chunk text",
        start_position=0,
        end_position=10,
        modality="text",
        metadata={"fixture_id": "kg-task20-synthetic-v1", "chunk_symbol": "chunk"},
        embedding=vector,
    )
    manifest = SimpleNamespace(
        fixture_id="kg-task20-synthetic-v1",
        documents={
            "document": SimpleNamespace(
                document_id="doc-id",
                collection_id=7,
                full_text_sha256=hashlib.sha256(b"document text").hexdigest(),
            )
        },
        chunks={
            "chunk": SimpleNamespace(
                chunk_id=11,
                document_symbol="document",
                chunk_number=0,
                start=0,
                end=10,
                content_sha256=hashlib.sha256(b"chunk text").hexdigest(),
                embedding_sha256=authority.canonical_embedding_sha256(vector),
            )
        },
    )

    authority.validate_live_fixture_rows(
        manifest=manifest,
        documents=(document,),
        chunks=(chunk,),
    )
    chunk.content = "drifted private text"
    with pytest.raises(RuntimeError, match="chunk content"):
        authority.validate_live_fixture_rows(
            manifest=manifest,
            documents=(document,),
            chunks=(chunk,),
        )
