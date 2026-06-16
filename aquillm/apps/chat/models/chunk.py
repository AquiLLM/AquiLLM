"""ConversationChunk model for storing chat-transcript chunks with embeddings.

Mirrors apps.documents.models.chunks.TextChunk, but scoped to a WSConversation
instead of a Document. Indexing lives in apps.chat.services.conversation_indexing;
hybrid vector + trigram retrieval + reranking lives in
apps.chat.services.conversation_search (which reuses the documents reranker).
"""
from __future__ import annotations

import structlog

from django.contrib.postgres.indexes import GinIndex
from django.db import models
from pgvector.django import HnswIndex, VectorField

from .conversation import WSConversation

logger = structlog.stdlib.get_logger(__name__)


class ConversationChunk(models.Model):
    """A turn-window chunk of a past conversation with an embedding for vector search.

    Conversation chunks are always text. The ``Modality`` enum / ``modality`` field
    exist solely so the documents reranker (``rerank_chunks``), which inspects
    ``chunk.modality``, can be reused unchanged.
    """

    class Modality(models.TextChoices):
        TEXT = "text", "Text"
        IMAGE = "image", "Image"

    conversation = models.ForeignKey(
        WSConversation, on_delete=models.CASCADE, related_name="chunks"
    )
    content = models.TextField()
    chunk_number = models.PositiveIntegerField()
    # Inclusive Message.sequence_number range this chunk covers (for linking + dedupe).
    start_sequence = models.PositiveIntegerField()
    end_sequence = models.PositiveIntegerField()
    modality = models.CharField(
        max_length=16, choices=Modality.choices, default=Modality.TEXT
    )
    metadata = models.JSONField(default=dict, blank=True)
    embedding = VectorField(dimensions=1024, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        app_label = "apps_chat"
        db_table = "aquillm_conversationchunk"
        ordering = ["conversation", "chunk_number"]
        constraints = [
            models.UniqueConstraint(
                fields=["conversation", "chunk_number"],
                name="unique_chunk_per_conversation",
            ),
        ]
        indexes = [
            models.Index(fields=["conversation"], name="convchunk_conversation_idx"),
            HnswIndex(
                name="convchunk_embedding_idx",
                fields=["embedding"],
                m=16,
                ef_construction=64,
                opclasses=["vector_l2_ops"],
            ),
            GinIndex(
                name="convchunk_content_trgm_idx",
                fields=["content"],
                opclasses=["gin_trgm_ops"],
            ),
        ]

    def __str__(self):
        return f"conv {self.conversation_id} chunk {self.chunk_number}"
