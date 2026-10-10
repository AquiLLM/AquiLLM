"""Refresh a finite keyset page of active legacy projections without providers."""

import json

from django.core.management.base import BaseCommand, CommandError
from django.db import connections, transaction
from django.db.models.functions import Length

from apps.documents.models import DESCENDED_FROM_DOCUMENT, TextChunk
from apps.documents.services.lexical_publication import publish_chunk_search_vectors
from apps.documents.services.representation_contracts import (
    DocumentRefV1,
    SourceScopeUnavailable,
)
from apps.documents.services.representation_store import (
    active_chunk_scope,
    freeze_active_representations,
)


class Command(BaseCommand):
    help = (
        "Refresh bounded active legacy text pages; skip other modalities/generations."
    )

    def add_arguments(self, parser):
        parser.add_argument("--database", default="default")
        parser.add_argument("--batch-size", type=int, default=100)
        parser.add_argument("--max-batches", type=int, default=1)
        parser.add_argument("--after-pk", type=int, default=0)

    def handle(self, *args, **options):
        using = options["database"]
        batch, maximum, after = (
            options[n] for n in ("batch_size", "max_batches", "after_pk")
        )
        for value, low, high in (
            (batch, 1, 1000),
            (maximum, 1, 100),
            (after, 0, 2**63 - 1),
        ):
            if type(value) is not int or not low <= value <= high:
                raise CommandError("invalid bounded backfill controls")
        if using not in connections:
            raise CommandError("unknown database")
        processed = skipped = batches = 0
        for _ in range(maximum):
            with transaction.atomic(using=using):
                with connections[using].cursor() as cursor:
                    cursor.execute(
                        "SELECT set_config('statement_timeout', "
                        "LEAST(COALESCE(NULLIF(setting::bigint, 0), 5000), "
                        "5000)::text, true) "
                        "FROM pg_settings WHERE name='statement_timeout'"
                    )
                # Apply the keyset LIMIT before concrete owner resolution or body reads.
                page = tuple(
                    TextChunk.objects.using(using)
                    .filter(pk__gt=after)
                    .order_by("pk")
                    .values_list("pk", "doc_id")[:batch]
                )
                if not page:
                    break
                for document_id in dict.fromkeys(doc_id for _, doc_id in page):
                    ids = [pk for pk, doc_id in page if doc_id == document_id]
                    owners = [
                        (model, pk)
                        for model in DESCENDED_FROM_DOCUMENT
                        for pk in model._base_manager.using(using)
                        .filter(id=document_id)
                        .values_list("pkid", flat=True)[:2]
                    ]
                    if len(owners) != 1:
                        skipped += len(ids)
                        continue
                    model, pkid = owners[0]
                    doc = (
                        model._base_manager.using(using)
                        .annotate(_size=Length("full_text"))
                        .filter(pkid=pkid, id=document_id, _size__lte=1_000_000)
                        .select_for_update()
                        .first()
                    )
                    if doc is None:
                        skipped += len(ids)
                        continue
                    try:
                        ref = DocumentRefV1(
                            doc._meta.label_lower,
                            doc.pkid,
                            doc.id,
                            doc.collection_id,
                            doc.full_text_hash,
                        )
                        snapshot = freeze_active_representations((ref,), using=using)
                        rows = tuple(
                            active_chunk_scope(
                                TextChunk.objects.using(using).filter(
                                    pk__in=ids,
                                    modality="text",
                                    representation_generation__isnull=True,
                                    source_kind="unknown",
                                    section_path=[],
                                ),
                                snapshot=snapshot,
                            ).select_for_update()
                        )
                        publish_chunk_search_vectors(
                            document=doc, chunks=rows, using=using
                        )
                    except SourceScopeUnavailable:
                        # Roll back the complete batch on source changes.
                        raise CommandError(
                            "source changed during bounded backfill"
                        ) from None
                    processed += len(rows)
                    skipped += len(ids) - len(rows)
                after = page[-1][0]
                batches += 1
        self.stdout.write(
            json.dumps(
                {
                    "processed": processed,
                    "skipped": skipped,
                    "after_pk": after,
                    "batches": batches,
                    "mode": "active-legacy-text",
                }
            )
        )
