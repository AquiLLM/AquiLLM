"""
TextChunk model for storing document chunks with embeddings.

Retrieval and reranking live in apps.documents.services.chunk_*.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import structlog
from django.contrib.postgres.indexes import GinIndex
from django.contrib.postgres.search import SearchVectorField
from django.core.exceptions import ValidationError
from django.db import models
from pgvector.django import HnswIndex, VectorField

from lib.embeddings.persistence import EmbeddingQuerySet, prepare_embedding_save

if TYPE_CHECKING:
    pass

logger = structlog.stdlib.get_logger(__name__)


class TextChunkQuerySet(EmbeddingQuerySet):
    """Custom QuerySet for TextChunk with document filtering."""

    def filter_by_documents(self, docs_or_ids):
        ids = [getattr(doc_or_id, "id", doc_or_id) for doc_or_id in docs_or_ids]
        return self.filter(doc_id__in=ids)


def _get_descended_from_document():
    """Lazy import to avoid circular dependency."""
    from .document_types import DESCENDED_FROM_DOCUMENT

    return DESCENDED_FROM_DOCUMENT


def doc_id_validator(id):
    descended_from_document = _get_descended_from_document()
    if sum([t.objects.filter(id=id).exists() for t in descended_from_document]) != 1:
        raise ValidationError(
            "Invalid Document UUID -- either no such document or multiple"
        )


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
    search_vector = SearchVectorField(null=True, editable=False)
    search_vector_binding = models.JSONField(null=True, editable=False)
    source_kind = models.CharField(
        max_length=32,
        default="unknown",
        choices=[
            (kind, kind)
            for kind in (
                "unknown",
                "body",
                "heading",
                "table",
                "formula",
                "figure",
                "bibliography",
            )
        ],
    )
    section_path = models.JSONField(default=list)
    embedding = VectorField(dimensions=1024, blank=True, null=True)
    embedding_provenance = models.JSONField(blank=True, null=True)
    representation_generation = models.ForeignKey(
        "DocumentRepresentationGeneration", null=True, on_delete=models.DO_NOTHING,
        db_column="representation_generation_id",
    )
    legacy_source_snapshot = models.ForeignKey(
        "LegacyDocumentSourceSnapshot", null=True, on_delete=models.DO_NOTHING,
    )
    legacy_context_status = models.CharField(max_length=16, default="unbound")
    legacy_artifact_status = models.CharField(max_length=16, default="unbound")
    legacy_context_failure_code = models.CharField(max_length=64, default="", blank=True)
    legacy_artifact_failure_code = models.CharField(max_length=64, default="", blank=True)

    doc_id = models.UUIDField(editable=False, validators=[doc_id_validator])

    objects = TextChunkQuerySet.as_manager()

    class Meta:
        app_label = "apps_documents"
        db_table = "aquillm_textchunk"
        constraints = [
            models.UniqueConstraint(
                fields=["doc_id", "start_position", "end_position"],
                name="legacy_chunk_position_unique",
                condition=models.Q(representation_generation__isnull=True),
            ),
            models.UniqueConstraint(
                fields=["doc_id", "chunk_number"],
                name="legacy_chunk_number_unique",
                condition=models.Q(representation_generation__isnull=True),
            ),
            models.UniqueConstraint(
                fields=["doc_id", "representation_generation", "chunk_number"],
                name="generation_chunk_number_unique",
            ),
            models.UniqueConstraint(
                fields=["doc_id", "representation_generation", "start_position", "end_position"],
                name="generation_chunk_position_unique",
            ),
            models.CheckConstraint(
                condition=models.Q(representation_generation__isnull=True) | models.Q(legacy_source_snapshot__isnull=True),
                name="chunk_single_source_binding",
            ),
        ]
        indexes = [
            GinIndex(fields=["search_vector"], name="textchunk_search_vector_gin_v1"),
            models.Index(fields=["doc_id", "start_position", "end_position"]),
            models.Index(
                fields=["doc_id", "modality"], name="textchunk_doc_modality_idx"
            ),
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
        from apps.documents.services.representation_history import exact_chunk_document
        return exact_chunk_document(self, using=self._state.db or "default")

    @document.setter
    def document(self, doc):
        self.doc_id = doc.id

    def save(self, *args, **kwargs):
        if self.start_position >= self.end_position:
            raise ValueError("end_position must be greater than start_position")
        fields = kwargs.get("update_fields")
        writes_embedding = fields is None or "embedding" in fields
        if writes_embedding and (self.embedding is None or len(self.embedding) == 0):
            try:
                self.get_chunk_embedding()
            except Exception:
                logger.warning(
                    "obs.documents.chunk_embed_failed",
                    reason="upstream_unavailable",
                )

        prepare_embedding_save(self, kwargs)
        super().save(*args, **kwargs)

    def get_chunk_embedding(self, callback: Callable[[], None] | None = None):
        from apps.documents.services.chunk_embeddings import (
            get_chunk_embedding as compute_embedding,
        )

        return compute_embedding(self, callback)

    @classmethod
    def rerank(cls, query: str, chunks, top_k: int):
        from apps.documents.services.chunk_rerank import rerank_chunks

        return rerank_chunks(cls, query, chunks, top_k)

    @classmethod
    def text_chunk_search(
        cls,
        query: str,
        top_k: int,
        docs: list,
        *,
        authorization_context: object | None = None,
        hybrid_graph_dependencies: object | None = None,
    ):
        from apps.documents.services.chunk_search import (
            text_chunk_search as hybrid_search,
        )

        return hybrid_search(
            cls,
            query,
            top_k,
            docs,
            authorization_context=authorization_context,
            hybrid_graph_dependencies=hybrid_graph_dependencies,
        )
