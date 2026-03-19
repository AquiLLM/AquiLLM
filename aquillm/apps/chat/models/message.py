"""Message model - stores individual conversation messages."""
import uuid

from django.db import models

from .conversation import WSConversation


class Message(models.Model):
    ROLE_CHOICES = [('user', 'User'), ('assistant', 'Assistant'), ('tool', 'Tool')]
    FOR_WHOM_CHOICES = [('user', 'User'), ('assistant', 'Assistant')]

    conversation = models.ForeignKey(WSConversation, on_delete=models.CASCADE, related_name='db_messages')
    message_uuid = models.UUIDField(default=uuid.uuid4, db_index=True)
    role = models.CharField(max_length=10, choices=ROLE_CHOICES)
    content = models.TextField()
    rating = models.PositiveSmallIntegerField(null=True, blank=True)
    feedback_text = models.TextField(null=True, blank=True)
    sequence_number = models.PositiveIntegerField()
    created_at = models.DateTimeField(auto_now_add=True)

    # AssistantMessage-specific fields
    model = models.CharField(max_length=100, null=True, blank=True)
    stop_reason = models.CharField(max_length=50, null=True, blank=True)
    tool_call_id = models.CharField(max_length=100, null=True, blank=True)
    tool_call_name = models.CharField(max_length=100, null=True, blank=True)
    tool_call_input = models.JSONField(null=True, blank=True)
    usage = models.PositiveIntegerField(default=0)

    # ToolMessage-specific fields
    tool_name = models.CharField(max_length=100, null=True, blank=True)
    arguments = models.JSONField(null=True, blank=True)
    for_whom = models.CharField(max_length=10, choices=FOR_WHOM_CHOICES, null=True, blank=True)
    result_dict = models.JSONField(null=True, blank=True)

    class Meta:
        app_label = 'apps_chat'
        db_table = 'aquillm_message'
        ordering = ['conversation', 'sequence_number']
        indexes = [models.Index(fields=['rating'])]
