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
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        app_label = 'apps_memory'
        db_table = 'aquillm_usermemoryfact'
        ordering = ['category', 'created_at']

    def __str__(self):
        return f"{self.user.username}: {self.fact[:50]}..."
