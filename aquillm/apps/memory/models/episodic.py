"""Episodic memory model."""
import structlog

from django.contrib.auth.models import User
from django.db import models
from django.db.models import Q
from pgvector.django import VectorField, HnswIndex

logger = structlog.stdlib.get_logger(__name__)


class EpisodicMemory(models.Model):
    """
    Embeddings of past conversation turns for semantic retrieval across threads.
    When the user sends a message, we retrieve top-k similar past exchanges and inject them into context.
    """
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='episodic_memories')
    content = models.TextField(help_text="Summary or excerpt of a past exchange (User: ... Assistant: ...)")
    embedding = VectorField(dimensions=1024, blank=True, null=True)
    conversation = models.ForeignKey(
        'apps_chat.WSConversation',
        on_delete=models.CASCADE,
        related_name='episodic_memories',
        null=True,
        blank=True,
    )
    assistant_message_uuid = models.UUIDField(
        null=True,
        blank=True,
        db_index=True,
        help_text="Dedupe: we only store one episodic memory per assistant message",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        app_label = 'apps_memory'
        db_table = 'aquillm_episodicmemory'
        ordering = ['-created_at']
        indexes = [
            HnswIndex(
                name='episodic_memory_embedding_idx',
                fields=['embedding'],
                m=16,
                ef_construction=64,
                opclasses=['vector_l2_ops'],
            ),
        ]
        constraints = [
            models.UniqueConstraint(
                fields=['user', 'assistant_message_uuid'],
                name='unique_episodic_per_assistant_msg',
                condition=Q(assistant_message_uuid__isnull=False),
            ),
        ]

    def __str__(self):
        return f"{self.user.username}: {self.content[:50]}..."

    def save(self, *args, **kwargs):
        if not self.embedding and self.content:
            try:
                from aquillm.utils import get_embedding
                self.embedding = get_embedding(self.content, input_type='search_document')
            except Exception as exc:
                logger.warning(
                    "obs.memory.episodic_warning",
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
        super().save(*args, **kwargs)
