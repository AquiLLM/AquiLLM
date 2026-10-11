"""Atomic projection publication inside the source owner's existing transaction."""

from django.contrib.postgres.search import SearchVector
from django.db import connections
from django.db.models import Value

from .lexical_projection import (
    LexicalProjectionV1,
    binding_expression,
    structural_projection,
    text_hash,
)
from .representation_contracts import DocumentRefV1, SourceScopeUnavailable


def publish_legacy_projections(document, chunks, *, using):
    for start in range(0, len(chunks), 100):
        publish_chunk_search_vectors(
            document=document, chunks=tuple(chunks[start : start + 100]), using=using
        )


def publish_chunk_search_vectors(
    *,
    document,
    chunks,
    using,
    canonical_text=None,
    representation_key=None,
    source_revision_sha256=None,
    extraction=None,
):
    from apps.documents.models import DESCENDED_FROM_DOCUMENT, TextChunk

    if (
        not connections[using].in_atomic_block
        or type(chunks) is not tuple
        or len(chunks) > 1000
    ):
        raise ValueError("bounded_lexical_publication_transaction_required")
    if len({c.pk for c in chunks}) != len(chunks):
        raise ValueError("duplicate_lexical_chunk")
    owners = []
    for model in DESCENDED_FROM_DOCUMENT:
        owners.extend(
            (model, row)
            for row in model._base_manager.using(using)
            .filter(id=document.id)
            .values("pkid", "collection_id")[:2]
        )
    if owners != [
        (
            type(document),
            {"pkid": document.pkid, "collection_id": document.collection_id},
        )
    ]:
        raise SourceScopeUnavailable("source_scope_unavailable")
    locked = (
        type(document)
        ._base_manager.using(using)
        .select_for_update()
        .get(pkid=document.pkid, id=document.id)
    )
    if (locked.full_text_hash, locked.title, locked.collection_id) != (
        document.full_text_hash,
        document.title,
        document.collection_id,
    ):
        raise SourceScopeUnavailable("source_scope_unavailable")
    if text_hash(locked.full_text) != locked.full_text_hash:
        raise SourceScopeUnavailable("source_scope_unavailable")
    canonical_text = locked.full_text if canonical_text is None else canonical_text
    if representation_key is None and canonical_text != locked.full_text:
        raise SourceScopeUnavailable("source_scope_unavailable")
    ref = DocumentRefV1(
        locked._meta.label_lower,
        locked.pkid,
        locked.id,
        locked.collection_id,
        locked.full_text_hash,
    )
    projection = LexicalProjectionV1(
        ref,
        text_hash(canonical_text),
        representation_key,
        source_revision_sha256,
        locked.title,
    )
    for chunk in chunks:
        if (
            type(chunk.pk) is not int
            or chunk.pk <= 0
            or chunk.doc_id != locked.id
            or chunk.representation_generation_id != representation_key
        ):
            raise SourceScopeUnavailable("source_scope_unavailable")
        if chunk.modality != TextChunk.Modality.TEXT:
            continue  # Explicitly unsupported; never manufacture text coverage.
        if (
            not 0 <= chunk.start_position < chunk.end_position <= len(canonical_text)
            or canonical_text[chunk.start_position : chunk.end_position]
            != chunk.content
        ):
            raise SourceScopeUnavailable("source_scope_unavailable")
        kind, section = structural_projection(chunk, canonical_text, extraction)
        rows = TextChunk.objects.using(using).filter(
            pk=chunk.pk,
            doc_id=locked.id,
            content=chunk.content,
            start_position=chunk.start_position,
            end_position=chunk.end_position,
            chunk_number=chunk.chunk_number,
            representation_generation=representation_key,
            modality=TextChunk.Modality.TEXT,
        )
        vector = (
            SearchVector(Value(locked.title), config="english", weight="A")
            + SearchVector(Value("\n".join(section)), config="english", weight="B")
            + SearchVector("content", config="english", weight="D")
        )
        if (
            rows.update(search_vector=vector, source_kind=kind, section_path=section)
            != 1
        ):
            raise SourceScopeUnavailable("source_scope_unavailable")
        rows.update(search_vector_binding=binding_expression(projection))
    return projection
