"""CollectionSkill model: per-collection markdown notes merged into the chat system prompt.

One row per collection. Body is the markdown "Collection Notes" content owners
write to teach AquiLLM domain-specific facts (author lists, terminology,
project conventions, etc.). Absence of a row means the collection has no notes
and behaves exactly as before.
"""
from django.contrib.auth.models import User
from django.db import models

from apps.collections.models import Collection


MAX_BODY_LENGTH = 16_000


class CollectionSkill(models.Model):
    collection = models.OneToOneField(
        Collection, on_delete=models.CASCADE, related_name="skill"
    )
    body = models.TextField()
    updated_by = models.ForeignKey(
        User, on_delete=models.SET_NULL, null=True, blank=True, related_name="+"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        app_label = "apps_skills"
        db_table = "aquillm_collectionskill"

    def __str__(self) -> str:
        return f"CollectionSkill({self.collection_id})"
