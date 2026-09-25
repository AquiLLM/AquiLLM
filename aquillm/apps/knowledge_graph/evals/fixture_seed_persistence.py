"""Transactional persistence for graph-row-free synthetic fixtures."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from django.db import transaction

from .fixture_manifest import assemble_fixture_document, canonical_embedding_sha256
from .fixture_seed_contract import (
    DIMENSIONS,
    FIXTURE_ID,
    HIDDEN_USERNAME,
    INPUT_TYPE,
    PHYSICAL_BINDINGS,
    PHYSICAL_LABELS,
    VISIBLE_USERNAME,
    EmbeddingIdentity,
    FixtureSeedError,
    LogicalFixture,
    document_ids,
    request_ids,
)
from .fixture_seed_manifest_io import canonical_manifest_bytes, validate_payload


def assert_no_identity_collisions(logical: LogicalFixture) -> None:
    from django.contrib.auth.models import User

    from apps.collections.models import Collection
    from apps.documents.models import DESCENDED_FROM_DOCUMENT, TextChunk
    from apps.knowledge_graph.models import GraphRebuildRequest

    names = [f"{FIXTURE_ID}-{label}" for label in PHYSICAL_LABELS]
    expected_documents = tuple(document_ids(logical).values())
    collision = (
        Collection.objects.filter(name__in=names).exists()
        or User.objects.filter(
            username__in=(VISIBLE_USERNAME, HIDDEN_USERNAME)
        ).exists()
        or TextChunk.objects.filter(doc_id__in=expected_documents).exists()
        or GraphRebuildRequest.objects.filter(pk__in=request_ids().values()).exists()
        or any(
            model.objects.filter(id__in=expected_documents).exists()
            for model in DESCENDED_FROM_DOCUMENT
        )
    )
    if collision:
        raise FixtureSeedError("deterministic fixture identity collision")


def _create_principals_and_collections():
    from django.contrib.auth.models import User

    from apps.collections.models import Collection, CollectionPermission

    users = {}
    for username in (VISIBLE_USERNAME, HIDDEN_USERNAME):
        user = User(
            username=username,
            email="",
            is_active=False,
            is_staff=False,
            is_superuser=False,
        )
        user.set_unusable_password()
        user.save(force_insert=True)
        users[username] = user
    collections = {
        label: Collection.objects.create(name=f"{FIXTURE_ID}-{label}")
        for label in PHYSICAL_LABELS
    }
    CollectionPermission.objects.bulk_create(
        [
            *(
                CollectionPermission(
                    user=users[VISIBLE_USERNAME],
                    collection=collections[label],
                    permission="MANAGE",
                )
                for label in PHYSICAL_LABELS
                if label != "hidden"
            ),
            CollectionPermission(
                user=users[HIDDEN_USERNAME],
                collection=collections["hidden"],
                permission="MANAGE",
            ),
        ]
    )
    return users, collections


def _create_documents(logical, users, collections):
    from apps.documents.models import RawTextDocument

    ids = document_ids(logical)
    documents = []
    assemblies = {}
    for symbol, logical_document in logical.documents.items():
        full_text, spans = assemble_fixture_document(
            tuple(chunk.text for chunk in logical_document.chunks)
        )
        assemblies[symbol] = (full_text, spans)
        hidden = logical_document.collection_symbol == "collection-security-private"
        documents.append(
            RawTextDocument(
                id=ids[symbol],
                title=logical_document.title,
                full_text=full_text,
                full_text_hash=sha256(full_text.encode()).hexdigest(),
                collection=collections[
                    PHYSICAL_BINDINGS[logical_document.collection_symbol]
                ],
                ingested_by=users[HIDDEN_USERNAME if hidden else VISIBLE_USERNAME],
                ingestion_complete=True,
            )
        )
    RawTextDocument.objects.bulk_create(documents)
    return ids, assemblies


def _create_chunks(logical, ids, assemblies, vectors):
    from apps.documents.models import TextChunk

    chunks = []
    symbols = []
    for document_symbol, logical_document in logical.documents.items():
        for number, logical_chunk in enumerate(logical_document.chunks):
            start, end = assemblies[document_symbol][1][number]
            chunks.append(
                TextChunk(
                    doc_id=ids[document_symbol],
                    content=logical_chunk.text,
                    start_position=start,
                    end_position=end,
                    start_time=None,
                    chunk_number=number,
                    modality=TextChunk.Modality.TEXT,
                    metadata={
                        "chunk_symbol": logical_chunk.symbol,
                        "fixture_id": FIXTURE_ID,
                    },
                    embedding=list(vectors[logical_chunk.symbol]),
                )
            )
            symbols.append(logical_chunk.symbol)
    TextChunk.objects.bulk_create(chunks)
    if any(type(chunk.pk) is not int or chunk.pk < 1 for chunk in chunks):
        raise FixtureSeedError("fixture chunk identity publication failed")
    return dict(zip(symbols, chunks, strict=True))


def _build_payload(logical, embedding, vectors, collections, ids, assemblies, chunks):
    requests = request_ids()
    collection_rows = {}
    for symbol in sorted(PHYSICAL_BINDINGS):
        label = PHYSICAL_BINDINGS[symbol]
        authorized = label != "hidden"
        collection_rows[symbol] = {
            "authorized": authorized,
            "collection_id": collections[label].pk,
            "rebuild_request_id": str(requests[label]) if authorized else None,
        }
    scope = sorted({(collections[label].pk, requests[label]) for label in requests})
    return {
        "authorized_scope": [
            {"collection_id": collection_id, "rebuild_request_id": str(request_id)}
            for collection_id, request_id in scope
        ],
        "canonical_identity_assertions": [
            {
                "expected_outcome": "automatic",
                "source_chunk_symbol": source,
                "target_chunk_symbol": target,
            }
            for source, target in logical.canonical
        ],
        "chunks": {
            symbol: {
                "chunk_id": chunks[symbol].pk,
                "chunk_number": chunks[symbol].chunk_number,
                "content_sha256": sha256(chunks[symbol].content.encode()).hexdigest(),
                "document_symbol": logical.chunks[symbol][0],
                "embedding_sha256": canonical_embedding_sha256(vectors[symbol]),
                "end": chunks[symbol].end_position,
                "start": chunks[symbol].start_position,
            }
            for symbol in sorted(chunks)
        },
        "collections": collection_rows,
        "documents": {
            symbol: {
                "collection_symbol": document.collection_symbol,
                "document_id": str(ids[symbol]),
                "full_text_sha256": sha256(assemblies[symbol][0].encode()).hexdigest(),
            }
            for symbol, document in logical.documents.items()
        },
        "embedding": {
            "checkpoint": embedding.checkpoint,
            "dimensions": DIMENSIONS,
            "endpoint_signature": embedding.endpoint_signature,
            "input_type": INPUT_TYPE,
            "model": embedding.model,
        },
        "fixture_checksum": logical.checksum,
        "fixture_id": FIXTURE_ID,
        "inaccessible_neighbor_assertions": [
            {"source_chunk_symbol": source, "target_chunk_symbol": target}
            for source, target in logical.inaccessible
        ],
        "schema_version": 1,
    }


def persist_fixture(
    path: Path,
    logical: LogicalFixture,
    embedding: EmbeddingIdentity,
    vectors,
    *,
    publish,
):
    published_payload = None
    try:
        with transaction.atomic():
            users, collections = _create_principals_and_collections()
            ids, assemblies = _create_documents(logical, users, collections)
            chunks = _create_chunks(logical, ids, assemblies, vectors)
            payload = _build_payload(
                logical, embedding, vectors, collections, ids, assemblies, chunks
            )
            validate_payload(payload, logical)
            resolved = publish(path, payload, logical)
            published_payload = payload
        return resolved
    except Exception:
        if published_payload is not None:
            try:
                if path.read_bytes() == canonical_manifest_bytes(published_payload):
                    path.unlink()
            except OSError:
                pass
        raise
