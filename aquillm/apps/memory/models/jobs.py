"""One durable, coalesced memory request per conversation."""
from django.db import models
from django.utils import timezone


class ConversationMemoryJob(models.Model):
    conversation = models.OneToOneField('apps_chat.WSConversation', on_delete=models.CASCADE, primary_key=True)
    desired_transcript_hash = models.CharField(max_length=64, blank=True, default='')
    completed_transcript_hash = models.CharField(max_length=64, blank=True, default='')
    state = models.CharField(max_length=16, default='pending')
    token = models.UUIDField(null=True, blank=True)
    lease_expires_at = models.DateTimeField(null=True, blank=True)
    next_attempt_at = models.DateTimeField(default=timezone.now, db_index=True)
    attempts = models.PositiveIntegerField(default=0)
    last_error = models.CharField(max_length=128, blank=True, default='')

    class Meta:
        app_label = 'apps_memory'


class ProfileMemoryReconciliationJob(models.Model):
    """Content-free durable deletion of known, owner-verified provider copies."""
    fact = models.OneToOneField('apps_memory.UserMemoryFact', on_delete=models.CASCADE, primary_key=True)
    state = models.CharField(max_length=16, default='pending')
    token = models.UUIDField(null=True, blank=True)
    attempts = models.PositiveIntegerField(default=0)
    next_attempt_at = models.DateTimeField(default=timezone.now, db_index=True)
    lease_expires_at = models.DateTimeField(null=True, blank=True)
    last_error = models.CharField(max_length=32, blank=True, default='')

    class Meta:
        app_label = 'apps_memory'
