"""Zotero connection model."""
from django.contrib.auth.models import User
from django.db import models


class ZoteroConnection(models.Model):
    """Stores Zotero OAuth credentials for a user"""
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='zotero_connection')
    api_key = models.CharField(max_length=255, help_text="Zotero API key from OAuth")
    zotero_user_id = models.CharField(max_length=100, help_text="Zotero user ID")
    connected_at = models.DateTimeField(auto_now_add=True)
    last_synced_at = models.DateTimeField(null=True, blank=True, help_text="Last time a sync was performed")

    class Meta:
        app_label = 'apps_integrations_zotero'
        db_table = 'aquillm_zoteroconnection'

    def __str__(self):
        return f"{self.user.username}'s Zotero connection (User ID: {self.zotero_user_id})"
