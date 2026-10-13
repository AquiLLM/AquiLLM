"""User memory fact model."""
from django.contrib.auth.models import User
from django.db import models


USER_MEMORY_CATEGORY_CHOICES = [
    ('tone', 'Tone / style'),
    ('goals', 'Goals'),
    ('project', 'Project context'),
    ('preference', 'Preference'),
    ('general', 'General'),
]


class UserMemoryFact(models.Model):
    """
    Stable facts and preferences per user, injected into system context on every chat.
    E.g. tone, goals, project context. Can be set manually (future UI) or extracted from conversations.
    """
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='memory_facts')
    fact = models.TextField(help_text="Short statement of fact or preference")
    category = models.CharField(
        max_length=20,
        choices=USER_MEMORY_CATEGORY_CHOICES,
        default='general',
        db_index=True,
    )
    origin = models.CharField(max_length=24, default='legacy_unknown')
    source_conversation_id = models.BigIntegerField(null=True, blank=True, db_index=True)
    source_message_uuid = models.UUIDField(null=True, blank=True, db_index=True)
    source_content_digest = models.CharField(max_length=64, blank=True, default='')
    source_span_start = models.PositiveIntegerField(null=True, blank=True)
    source_span_end = models.PositiveIntegerField(null=True, blank=True)
    intent_policy_version = models.CharField(max_length=48, blank=True, default='')
    promotion_key = models.CharField(max_length=64, null=True, blank=True, unique=True)
    fact_digest = models.CharField(max_length=64, blank=True, default='')
    revision = models.PositiveIntegerField(default=1)
    status = models.CharField(max_length=16, default='active', db_index=True)
    supersedes = models.ForeignKey('self', on_delete=models.SET_NULL, null=True, blank=True, related_name='successors')
    provider_memory_ids = models.JSONField(default=list, blank=True)
    provider_receipt_complete = models.BooleanField(default=False)
    reconciliation_status = models.CharField(max_length=16, default='not_needed')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        app_label = 'apps_memory'
        db_table = 'aquillm_usermemoryfact'
        ordering = ['category', 'created_at']

    def __str__(self):
        return f"Profile fact {self.pk} ({self.status})"
