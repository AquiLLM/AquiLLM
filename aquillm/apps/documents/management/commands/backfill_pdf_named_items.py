"""Build one bounded keyset page of owner-locked PDF search projections."""

import json

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import connections, transaction
from django.db.models.functions import Length

from apps.documents.models import PDFDocument
from apps.documents.services.named_item_projection import (
    MAX_CANONICAL,
    publish_named_items,
)
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.services.representation_history import reference
from apps.documents.services.representation_store import freeze_active_representations


class Command(BaseCommand):
    help = "Build one finite page of current PDF item indexes and return its resume cursor."

    def add_arguments(self, parser):
        parser.add_argument("--after-pk", type=int, default=0)
        parser.add_argument("--batch-size", type=int, default=10)

    def handle(self, *args, **options):
        if not settings.RAG_NAMED_ITEM_INDEX_ENABLED:
            raise CommandError("PDF named item indexing is disabled")
        batch, after = options["batch_size"], options["after_pk"]
        if (
            type(batch) is not int
            or not 1 <= batch <= 100
            or type(after) is not int
            or not 0 <= after < 2**63
        ):
            raise CommandError("invalid bounded backfill controls")
        using = "default"
        documents = PDFDocument._base_manager.using(using)
        page = list(
            documents.filter(pk__gt=after)
            .order_by("pk")
            .values_list("pk", flat=True)[:batch]
        )
        ready = 0
        for pk in page:
            try:
                with transaction.atomic(using=using):
                    with connections[using].cursor() as cursor:
                        cursor.execute(
                            "SELECT set_config('statement_timeout', "
                            "LEAST(COALESCE(NULLIF(setting::bigint, 0), 5000), 5000)::text, true) "
                            "FROM pg_settings WHERE name='statement_timeout'"
                        )
                    doc = (
                        documents.annotate(size=Length("full_text"))
                        .filter(pk=pk, size__lte=MAX_CANONICAL)
                        .select_for_update()
                        .first()
                    )
                    if doc is not None:
                        (selected,) = freeze_active_representations(
                            (reference(doc),), using=using
                        ).selections
                        ready += publish_named_items(doc, selected, using=using)
            except SourceScopeUnavailable:
                # A concurrently changed owner has no verified current projection.
                continue
        self.stdout.write(
            json.dumps(
                {
                    "after_pk": page[-1] if page else after,
                    "ready": ready,
                    "skipped": len(page) - ready,
                }
            )
        )
