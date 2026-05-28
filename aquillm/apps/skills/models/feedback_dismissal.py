"""FeedbackDismissal: a manager has reviewed a piece of corrective feedback for
this collection and decided it doesn't warrant a notes update. Distinct from
dismissing a particular suggestion draft — that just throws out the draft and
lets the feedback be re-drafted. Dismissing the feedback row itself marks it
"handled" so the manager's queue stays tidy.
"""
from django.contrib.auth.models import User
from django.db import models

from apps.chat.models.message import Message
from apps.collections.models import Collection


class FeedbackDismissal(models.Model):
    collection = models.ForeignKey(
        Collection, on_delete=models.CASCADE, related_name="feedback_dismissals"
    )
    source_message = models.ForeignKey(
        Message, on_delete=models.CASCADE, related_name="feedback_dismissals"
    )
    dismissed_by = models.ForeignKey(
        User, on_delete=models.SET_NULL, null=True, blank=True, related_name="+"
    )
    dismissed_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        app_label = "apps_skills"
        db_table = "aquillm_feedbackdismissal"
        constraints = [
            models.UniqueConstraint(
                fields=["collection", "source_message"],
                name="uniq_feedback_dismissal_per_collection_message",
            )
        ]
        indexes = [models.Index(fields=["collection", "source_message"])]

    def __str__(self) -> str:
        return f"FeedbackDismissal(collection={self.collection_id}, msg={self.source_message_id})"
