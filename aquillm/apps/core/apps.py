from django.apps import AppConfig


class CoreConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.core'
    label = 'apps_core'

    def ready(self):
        from . import licensing_checks  # noqa: F401
