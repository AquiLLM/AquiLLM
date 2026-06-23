"""Notes app configuration."""
from django.apps import AppConfig


class NotesConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.notes"
    label = "apps_notes"
    verbose_name = "Collection Notes"
