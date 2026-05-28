"""SkillEditSuggestion: an LLM-drafted proposed edit to a collection's notes.

Generated on-demand when a collection manager clicks "Draft a suggestion" on
a piece of corrective user feedback (a low-rated assistant message with a
non-empty comment). Stays in PENDING status until a manager accepts or
dismisses it. AquiLLM never edits the notes file directly — every change
goes through manager approval.
"""
from django.contrib.auth.models import User
from django.db import models

from apps.chat.models.message import Message
from apps.collections.models import Collection


class SkillEditSuggestion(models.Model):
    STATUS_PENDING = "pending"
    STATUS_ACCEPTED = "accepted"
    STATUS_DISMISSED = "dismissed"
    STATUS_CHOICES = [
        (STATUS_PENDING, "Pending"),
        (STATUS_ACCEPTED, "Accepted"),
        (STATUS_DISMISSED, "Dismissed"),
    ]

    collection = models.ForeignKey(
        Collection, on_delete=models.CASCADE, related_name="skill_suggestions"
    )
    source_message = models.ForeignKey(
        Message, on_delete=models.CASCADE, related_name="skill_suggestions"
    )
    # Snapshot of the notes body at draft time; lets us show a faithful diff
    # even if the notes have been edited since the suggestion was generated.
    notes_body_at_generation = models.TextField()
    proposed_body = models.TextField()
    status = models.CharField(
        max_length=16, choices=STATUS_CHOICES, default=STATUS_PENDING, db_index=True
    )
    generated_by = models.ForeignKey(
        User, on_delete=models.SET_NULL, null=True, blank=True, related_name="+"
    )
    resolved_by = models.ForeignKey(
        User, on_delete=models.SET_NULL, null=True, blank=True, related_name="+"
    )
    resolved_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        app_label = "apps_skills"
        db_table = "aquillm_skilleditsuggestion"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["collection", "status"]),
            models.Index(fields=["source_message", "status"]),
        ]

    def __str__(self) -> str:
        return f"SkillEditSuggestion(collection={self.collection_id}, status={self.status})"
