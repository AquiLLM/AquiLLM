"""Bug report model."""
from django.contrib.auth.models import User
from django.db import models


SOURCE_CHOICES = (
    ('user', 'User'),
    ('exception', 'Exception'),
)


class BugReport(models.Model):
    user = models.ForeignKey(
        User, on_delete=models.SET_NULL, null=True, blank=True,
        related_name='bug_reports',
    )
    title = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    url = models.CharField(max_length=2048, blank=True)
    activity_log = models.JSONField(default=list)
    user_agent = models.CharField(max_length=512, blank=True)
    source = models.CharField(max_length=20, choices=SOURCE_CHOICES, default='user')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        app_label = 'apps_bug_reports'
        db_table = 'aquillm_bugreport'
        ordering = ['-created_at']

    def __str__(self):
        return f"[{self.source}] {self.title}"
