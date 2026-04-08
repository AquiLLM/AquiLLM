"""
TextChunk model for storing document chunks with embeddings.

Vector search, trigram similarity, and reranking live in apps.documents.services.chunk_*.
"""
from __future__ import annotations

import structlog
from typing import TYPE_CHECKING, Callable, List, Optional

from django.contrib.postgres.indexes import GinIndex
from django.core.exceptions import ValidationError
from django.db import models
from pgvector.django import HnswIndex, VectorField

if TYPE_CHECKING:
    from django.db.models.query import QuerySet

logger = structlog.stdlib.get_logger(__name__)


class TextChunkQuerySet(models.QuerySet):
    """Custom QuerySet for TextChunk with document filtering."""

    def filter_by_documents(self, docs_or_ids):
        ids = [
            getattr(doc_or_id, "id", doc_or_id)
            for doc_or_id in docs_or_ids
        ]
        return self.filter(doc_id__in=ids)


def _get_descended_from_document():
    """Lazy import to avoid circular dependency."""
    from .document_types import DESCENDED_FROM_DOCUMENT
    return DESCENDED_FROM_DOCUMENT


def doc_id_validator(id):
    descended_from_document = _get_descended_from_document()
    if sum([t.objects.filter(id=id).exists() for t in descended_from_document]) != 1:
        raise ValidationError("Invalid Document UUID -- either no such document or multiple")


class TextChunk(models.Model):
    """
    A chunk of text (or image caption) from a document with embedding for vector search.

    Supports both text and image modalities for multimodal RAG.
    """

    class Modality(models.TextChoices):
        TEXT = "text", "Text"
        IMAGE = "image", "Image"

    content = models.TextField()
    start_position = models.PositiveIntegerField()
    end_position = models.PositiveIntegerField()

    start_time = models.FloatField(null=True)
    chunk_number = models.PositiveIntegerField()
    modality = models.CharField(
        max_length=16, choices=Modality.choices, default=Modality.TEXT, db_index=True
    )
    metadata = models.JSONField(default=dict, blank=True)
    embedding = VectorField(dimensions=1024, blank=True, null=True)

    doc_id = models.UUIDField(editable=False, validators=[doc_id_validator])

    objects = TextChunkQuerySet.as_manager()

    class Meta:
        app_label = "apps_documents"
        db_table = "aquillm_textchunk"
        constraints = [
            models.UniqueConstraint(
                fields=["doc_id", "start_position", "end_position"],
                name="unique_chunk_position_per_document",
            ),
            models.UniqueConstraint(
                fields=["doc_id", "chunk_number"],
                name="uniqe_chunk_per_document",
            ),
        ]
        indexes = [
            models.Index(fields=["doc_id", "start_position", "end_position"]),
            models.Index(fields=["doc_id", "modality"], name="textchunk_doc_modality_idx"),
            HnswIndex(
                name="chunk_embedding_index",
                fields=["embedding"],
                m=16,
                ef_construction=64,
                opclasses=["vector_l2_ops"],
            ),
            GinIndex(
                name="textchunk_content_trgm_idx",
                fields=["content"],
                opclasses=["gin_trgm_ops"],
            ),
        ]
        ordering = ["doc_id", "chunk_number"]

    @property
    def document(self):
        """Get the parent document for this chunk."""
        descended_from_document = _get_descended_from_document()
        ret = None
        for t in descended_from_document:
            doc = t.objects.filter(id=self.doc_id).first()
            if doc:
                ret = doc
        if not ret:
            raise ValidationError(f"TextChunk {self.pk} is not associated with a document!")
        return ret

    @document.setter
    def document(self, doc):
        self.doc_id = doc.id

    def save(self, *args, **kwargs):
        if self.start_position >= self.end_position:
            raise ValueError("end_position must be greater than start_position")
        if not self.embedding:
            try:
                self.get_chunk_embedding()
            except Exception as exc:
                logger.warning(
                    "obs.documents.chunk_warning",
                    doc_id=str(self.doc_id),
                    chunk_number=self.chunk_number,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )

        super().save(*args, **kwargs)

    def get_chunk_embedding(self, callback: Optional[Callable[[], None]] = None):
        from apps.documents.services.chunk_embeddings import get_chunk_embedding as compute_embedding

        return compute_embedding(self, callback)

    @classmethod
    def rerank(cls, query: str, chunks, top_k: int):
        from apps.documents.services.chunk_rerank import rerank_chunks

        return rerank_chunks(cls, query, chunks, top_k)

    @classmethod
    def text_chunk_search(cls, query: str, top_k: int, docs: List):
        from apps.documents.services.chunk_search import text_chunk_search as hybrid_search

        return hybrid_search(cls, query, top_k, docs)
