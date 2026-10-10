from django.apps import AppConfig


class MemoryConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.memory'
    label = 'apps_memory'

    def ready(self):
        from . import signals  # noqa: F401
