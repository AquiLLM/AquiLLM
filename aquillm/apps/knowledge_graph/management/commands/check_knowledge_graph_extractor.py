from __future__ import annotations

from django.core.management.base import BaseCommand, CommandError

_FIXTURE = "Qwen3 uses MMLU."


class Command(BaseCommand):
    help = "Explicitly load and validate the configured graph extractor checkpoint."

    def handle(self, *args, **options):
        # This is deliberately the only operations command that imports the
        # optional provider factory, and it does so only after explicit invocation.
        try:
            from apps.knowledge_graph.services.builds import _active_ontology
            from lib.knowledge_graph.config import load_extraction_settings
            from lib.knowledge_graph.extractors import get_extraction_backend
            from lib.knowledge_graph.extractors.gliner2_local import GLINER2_VERSION

            settings = load_extraction_settings()
            ontology = _active_ontology()
            backend = get_extraction_backend(settings=settings)
            results = backend.extract_batch((_FIXTURE,), ontology=ontology)
        except Exception as exc:
            raise CommandError(f"extractor check failed: {type(exc).__name__}") from exc
        if len(results) != 1:
            raise CommandError("extractor check returned an invalid result count")
        result = results[0]
        if len(result.entities) < 2 or not result.relations:
            raise CommandError(
                "extractor fixture did not produce entity/relation evidence"
            )
        spans = tuple(
            (entity.start, entity.end, entity.text) for entity in result.entities
        )
        if any(_FIXTURE[start:end] != text for start, end, text in spans):
            raise CommandError("extractor fixture returned invalid entity spans")
        for relation in result.relations:
            if (
                _FIXTURE[relation.head_start : relation.head_end] != relation.head_text
                or _FIXTURE[relation.tail_start : relation.tail_end]
                != relation.tail_text
            ):
                raise CommandError("extractor fixture returned invalid relation spans")
        self.stdout.write(
            f"provider={settings.provider} package=gliner2=={GLINER2_VERSION} "
            f"model={settings.model_id} "
            f"revision={settings.model_revision}"
        )
