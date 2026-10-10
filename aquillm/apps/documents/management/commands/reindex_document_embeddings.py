"""Explicit vector-only rebuild of one exact document, never re-chunking."""

from uuid import UUID
from django.core.management.base import BaseCommand, CommandError
from django.apps import apps
from django.core.exceptions import ObjectDoesNotExist

from apps.documents.models import DESCENDED_FROM_DOCUMENT
from apps.documents.services.representation_contracts import DocumentRefV1
from apps.documents.services.representation_store import freeze_active_representations
from apps.documents.tasks.embedding_index import rebuild_document_embeddings


class Command(BaseCommand):
    help = "Schedule a verified vector-only generation for one exact document."

    def add_arguments(self, parser):
        parser.add_argument("--model-label", required=True)
        parser.add_argument("--document-pkid", required=True, type=int)
        parser.add_argument("--document-id", required=True, type=UUID)
        parser.add_argument("--space-key", required=True)
        parser.add_argument("--database", default="default")
        parser.add_argument("--timeout-ms", type=int, default=120000)

    def handle(self, **options):
        alias = options["database"]
        try:
            model = apps.get_model(options["model_label"])
            if model not in DESCENDED_FROM_DOCUMENT:
                raise ValueError()
            doc = model.objects.using(alias).get(
                pk=options["document_pkid"], id=options["document_id"]
            )
            ref = DocumentRefV1(
                model._meta.label_lower,
                doc.pkid,
                doc.id,
                doc.collection_id,
                doc.full_text_hash,
            )
            snapshot = freeze_active_representations((ref,), using=alias)
            key = rebuild_document_embeddings(
                snapshot.selections[0],
                space_key=options["space_key"],
                using=alias,
                timeout_ms=options["timeout_ms"],
            )
        except (LookupError, ValueError, ObjectDoesNotExist) as exc:
            raise CommandError(
                "Exact document or verified embedding scope unavailable"
            ) from exc
        self.stdout.write(key)
