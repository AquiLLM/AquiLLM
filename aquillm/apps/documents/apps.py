from django.apps import AppConfig


class DocumentsConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.documents"
    label = "apps_documents"

    def ready(self):
        from django.db.models.signals import post_delete
        from .models import DESCENDED_FROM_DOCUMENT
        from .signals import delete_document_chunks

        for model in DESCENDED_FROM_DOCUMENT:
            post_delete.connect(
                delete_document_chunks,
                sender=model,
                dispatch_uid=f"documents.delete_chunks.{model._meta.label_lower}",
            )
