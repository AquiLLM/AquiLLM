from django.apps import AppConfig


class KnowledgeGraphConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.knowledge_graph"
    label = "apps_knowledge_graph"

    def ready(self) -> None:
        from apps.collections.models import Collection
        from apps.documents.models import DESCENDED_FROM_DOCUMENT
        from apps.knowledge_graph.graph.invalidation import (
            register_collection_lifecycle_signals,
            register_document_lifecycle_signals,
        )
        from apps.knowledge_graph.resolution.canonical import (
            register_canonical_lifecycle_signals,
        )

        register_document_lifecycle_signals(DESCENDED_FROM_DOCUMENT)
        register_collection_lifecycle_signals(Collection)
        register_canonical_lifecycle_signals()
