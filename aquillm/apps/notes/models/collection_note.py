"""CollectionNote model: per-collection markdown notes injected into chat.

One row per collection. Body is the markdown "Collection Notes" content owners
write to teach AquiLLM domain-specific facts (author lists, terminology,
project conventions, etc.). Absence of a row means the collection has no notes
and behaves exactly as before. The notes are surfaced to the LLM inside the
document search tool results (see
``apps/chat/services/tool_wiring/documents.py``), not the system prompt.
"""
from django.contrib.auth.models import User
from django.db import models

from apps.collections.models import Collection


MAX_BODY_LENGTH = 16_000


class CollectionNote(models.Model):
    collection = models.OneToOneField(
        Collection, on_delete=models.CASCADE, related_name="note"
    )
    body = models.TextField()
    updated_by = models.ForeignKey(
        User, on_delete=models.SET_NULL, null=True, blank=True, related_name="+"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        app_label = "apps_notes"
        db_table = "aquillm_collectionnote"

    def __str__(self) -> str:
        return f"CollectionNote({self.collection_id})"
