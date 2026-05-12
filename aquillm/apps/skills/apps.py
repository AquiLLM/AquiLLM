"""Skills app configuration."""
from django.apps import AppConfig


class SkillsConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.skills"
    label = "apps_skills"
    verbose_name = "Skills"
