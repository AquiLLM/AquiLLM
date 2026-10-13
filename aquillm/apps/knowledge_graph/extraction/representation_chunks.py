"""Bounded extraction chunks from one exact active representation."""

from itertools import islice

from .evidence import ExtractionCapacityCode, ExtractionCapacityError


def ordered_chunks(document_id, *, for_update=False, representation_snapshot=None):
    from django.db.models import Sum
    from django.db.models.functions import Coalesce, Length

    from apps.documents.models import TextChunk
    from apps.knowledge_graph.projection.representation_inputs import (
        active_build_chunks,
    )

    from .pipeline import (
        _QUERY_ITERATOR_BATCH_SIZE,
        DOCUMENT_EXTRACTION_V1_MAX_CHARACTERS,
        DOCUMENT_EXTRACTION_V1_MAX_CHUNKS,
        StaleSourceError,
    )

    queryset = (
        TextChunk.objects.filter(doc_id=document_id)
        .only(
            "pk",
            "doc_id",
            "chunk_number",
            "start_position",
            "end_position",
            "modality",
            "content",
        )
        .order_by("chunk_number", "pk")
    )
    queryset = active_build_chunks(
        queryset, (document_id,), snapshot=representation_snapshot
    )
    if for_update:
        queryset = queryset.select_for_update()
    if queryset.count() > DOCUMENT_EXTRACTION_V1_MAX_CHUNKS:
        raise ExtractionCapacityError(
            ExtractionCapacityCode.CHUNK_LIMIT,
            "ordered document chunk cap exceeded",
        )
    totals = queryset.aggregate(total_characters=Coalesce(Sum(Length("content")), 0))
    total_characters = totals.get("total_characters")
    if type(total_characters) is not int or total_characters < 0:
        raise StaleSourceError("ordered document chunk character count is invalid")
    if total_characters > DOCUMENT_EXTRACTION_V1_MAX_CHARACTERS:
        raise ExtractionCapacityError(
            ExtractionCapacityCode.CHARACTER_LIMIT,
            "ordered document chunk character cap exceeded",
        )
    chunks = tuple(
        islice(
            queryset.iterator(
                chunk_size=min(
                    DOCUMENT_EXTRACTION_V1_MAX_CHUNKS,
                    _QUERY_ITERATOR_BATCH_SIZE,
                )
            ),
            DOCUMENT_EXTRACTION_V1_MAX_CHUNKS + 1,
        )
    )
    if len(chunks) > DOCUMENT_EXTRACTION_V1_MAX_CHUNKS:
        raise ExtractionCapacityError(
            ExtractionCapacityCode.CHUNK_LIMIT,
            "ordered document chunk cap exceeded",
        )
    return chunks
