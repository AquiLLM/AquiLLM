"""Durable receipts prevent replay of a tool whose outcome is uncertain."""

from django.db import models


class ToolExecution(models.Model):
    conversation = models.ForeignKey(
        "apps_chat.WSConversation", on_delete=models.CASCADE
    )
    call_id = models.CharField(max_length=255)
    fingerprint = models.CharField(max_length=64)
    result = models.JSONField(null=True)
    started_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        app_label = "apps_chat"
        constraints = [
            models.UniqueConstraint(
                fields=["conversation", "call_id"], name="chat_tool_execution_identity"
            )
        ]
