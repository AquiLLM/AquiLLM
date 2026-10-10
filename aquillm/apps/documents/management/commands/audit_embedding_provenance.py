"""Bounded, read-only counts of stored receipt coverage and vector integrity."""

import json

from django.core.management.base import BaseCommand, CommandError

from lib.embeddings.provenance import valid_provenance


class Command(BaseCommand):
    help = __doc__

    def add_arguments(self, parser):
        parser.add_argument("--limit", type=int, default=1000)
        parser.add_argument("--database", default="default")

    def handle(self, *args, **options):
        from apps.documents.models import TextChunk
        from apps.chat.models import ConversationChunk

        limit = options["limit"]
        if not 1 <= limit <= 10000:
            raise CommandError("limit must be between 1 and 10000")
        report = {"schema_version": 1, "limit_per_model": limit}
        for label, model in (
            ("documents", TextChunk),
            ("conversations", ConversationChunk),
        ):
            counts = dict(
                scanned=0,
                known=0,
                unknown=0,
                invalid=0,
                missing_vector=0,
                has_more=False,
            )
            rows = (
                model.objects.using(options["database"])
                .order_by("pk")
                .values_list("embedding", "embedding_provenance")[: limit + 1]
            )
            for vector, receipt in rows.iterator(chunk_size=100):
                if counts["scanned"] == limit:
                    counts["has_more"] = True
                    break
                counts["scanned"] += 1
                if vector is None:
                    counts["missing_vector"] += 1
                if receipt is None:
                    counts["unknown"] += 1
                elif valid_provenance(vector, receipt) is None:
                    counts["invalid"] += 1
                else:
                    counts["known"] += 1
            report[label] = counts
        self.stdout.write(json.dumps(report, sort_keys=True))
