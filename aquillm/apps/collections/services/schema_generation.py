# ruff: noqa: E402, I001 - support imports follow shared sample types
"""Bounded, local-only collection schema proposal helpers.

This module intentionally keeps collection text and inference output in local
variables.  Callers receive only canonical definitions and aggregate evidence.
"""

from __future__ import annotations

import heapq
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import islice

from .schema_generation_config import (
    SchemaGenerationConfig,
    SchemaGenerationConfigurationError,
    _enabled_from_environment as _enabled_from_environment,
    _positive_env_int as _positive_env_int,
    load_schema_generation_config,
)
from .schema_generation_sources import (
    _collection_source_documents as _collection_source_documents,
    _completed_collection_documents as _completed_collection_documents,
    _locked_collection_source_signature as _locked_collection_source_signature,
    collection_ingestion_pending as collection_ingestion_pending,
    collection_source_signature as collection_source_signature,
)

_MAX_ENTITY_TYPES = 24
_MIN_ENTITY_TYPES = 2
_MAX_RELATION_TYPES = 32
_MIN_RELATION_TYPES = 1


class InvalidSchemaCandidate(ValueError):
    """A model proposal cannot become a bounded collection draft."""


@dataclass(frozen=True, slots=True)
class SchemaSample:
    document_id: str
    chunk_id: int
    chunk_number: int
    text: str






def _eligible_collection_documents(collection_id: int) -> Iterator[dict[str, object]]:
    """Stream only completed documents with at least one usable text chunk."""

    from django.apps import apps
    from django.db.models import Exists, OuterRef

    from apps.documents.models.chunks import TextChunk

    usable_chunks = TextChunk.objects.filter(
        doc_id=OuterRef("id"),
        modality=TextChunk.Modality.TEXT,
        content__regex=r"\S",
    )
    iterators = []
    for name in (
        "PDFDocument",
        "TeXDocument",
        "RawTextDocument",
        "VTTDocument",
        "HandwrittenNotesDocument",
        "ImageUploadDocument",
        "MediaUploadDocument",
        "DocumentFigure",
    ):
        model = apps.get_model("apps_documents", name)
        queryset = (
            model.objects.filter(collection_id=collection_id, ingestion_complete=True)
            .annotate(has_usable_text=Exists(usable_chunks))
            .filter(has_usable_text=True)
            .order_by("id")
        )
        iterators.append(queryset.values("id").iterator())
    yield from heapq.merge(*iterators, key=lambda record: str(record["id"]))


def collection_has_eligible_text(collection_id: int) -> bool:
    """Return whether schema generation has any completed, usable source text."""

    return next(_eligible_collection_documents(collection_id), None) is not None








def _next_sample_chunk(
    document_id: str, after_chunk_number: int, character_limit: int
) -> SchemaSample | None:
    """Materialize one deterministic chunk, truncated before it leaves PostgreSQL."""

    from django.db.models.functions import Substr

    from apps.documents.models.chunks import TextChunk

    row = (
        TextChunk.objects.filter(
            doc_id=document_id,
            modality=TextChunk.Modality.TEXT,
            chunk_number__gt=after_chunk_number,
            content__regex=r"\S",
        )
        .order_by("chunk_number", "pk")
        .annotate(sample_content=Substr("content", 1, character_limit))
        .values("pk", "doc_id", "chunk_number", "sample_content")
        .first()
    )
    if row is None:
        return None
    return SchemaSample(
        str(row["doc_id"]),
        int(row["pk"]),
        int(row["chunk_number"]),
        row["sample_content"],
    )


def sample_collection_chunks(collection_id, max_chunks, max_characters):
    """Select deterministic, round-robin text chunks without exceeding either cap."""

    if type(max_chunks) is not int or max_chunks <= 0:
        raise ValueError("max_chunks must be a positive integer")
    if type(max_characters) is not int or max_characters <= 0:
        raise ValueError("max_characters must be a positive integer")
    selected: list[SchemaSample] = []
    document_ids = [
        str(row["id"])
        for row in islice(_eligible_collection_documents(collection_id), max_chunks)
    ]
    if not document_ids:
        return []
    cursors = {document_id: -1 for document_id in document_ids}
    active_documents = list(document_ids)
    used_characters = 0
    while (
        active_documents
        and len(selected) < max_chunks
        and used_characters < max_characters
    ):
        for document_id in tuple(active_documents):
            if len(selected) >= max_chunks or used_characters >= max_characters:
                break
            sample = _next_sample_chunk(
                document_id, cursors[document_id], max_characters - used_characters
            )
            if sample is None:
                active_documents.remove(document_id)
                cursors.pop(document_id)
                continue
            selected.append(sample)
            cursors[document_id] = sample.chunk_number
            used_characters += len(sample.text)
    return selected


def balanced_samples(
    groups: dict[str, list[SchemaSample]], *, max_chunks: int, max_characters: int
) -> list[SchemaSample]:
    """Round-robin ordered document groups and truncate only at the hard cap."""

    selected: list[SchemaSample] = []
    used_characters = 0
    offsets = {document_id: 0 for document_id in sorted(groups)}
    while len(selected) < max_chunks and used_characters < max_characters:
        progressed = False
        for document_id in sorted(groups):
            offset = offsets[document_id]
            if offset >= len(groups[document_id]) or len(selected) >= max_chunks:
                continue
            remaining = max_characters - used_characters
            if remaining <= 0:
                break
            sample = groups[document_id][offset]
            offsets[document_id] = offset + 1
            selected.append(
                SchemaSample(
                    sample.document_id,
                    sample.chunk_id,
                    sample.chunk_number,
                    sample.text[:remaining],
                )
            )
            used_characters += len(selected[-1].text)
            progressed = True
            if used_characters >= max_characters:
                break
        if not progressed:
            break
    return selected


from .schema_generation_support import (
    collect_candidate_evidence,
    generate_schema_candidate,
    normalize_schema_candidate,
)

__all__ = [
    "InvalidSchemaCandidate",
    "SchemaGenerationConfig",
    "SchemaGenerationConfigurationError",
    "SchemaSample",
    "collect_candidate_evidence",
    "collection_source_signature",
    "collection_ingestion_pending",
    "generate_schema_candidate",
    "load_schema_generation_config",
    "normalize_schema_candidate",
    "balanced_samples",
    "sample_collection_chunks",
]
