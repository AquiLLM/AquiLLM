"""Bounded vector-only rebuilds over exact current document representations."""
from collections import Counter
import json

from django.core.exceptions import ObjectDoesNotExist
from django.core.management.base import BaseCommand, CommandError
from django.db import DatabaseError

from apps.documents.services.embedding_index_deadline import (
    check_embedding_deadline, embedding_build_database,
)
from apps.documents.services.embedding_index_store import load_embedding_space
from apps.documents.services.embedding_reindex import plan_embedding_reindex, reindex_scope
from apps.documents.tasks.embedding_index import rebuild_document_embeddings


class Command(BaseCommand):
    help = ("Plan or schedule verified vector-only generations for exact owners or a bounded "
            "collection inventory. Dry-run reads metadata only; it does not certify input "
            "integrity or dense coverage and never calls providers or changes generations.")

    def add_arguments(self, parser):
        parser.add_argument("--model-label", required=True)
        parser.add_argument("--document-pkid", type=int)
        parser.add_argument("--document-id")
        parser.add_argument("--collection-id", type=int)
        parser.add_argument("--limit", type=int)
        parser.add_argument("--after-pkid", type=int)
        parser.add_argument("--space-key", required=True)
        parser.add_argument("--database", default="default")
        parser.add_argument("--timeout-ms", type=int, default=120000)
        parser.add_argument("--dry-run", action="store_true")

    def handle(self, **options):
        scope = reindex_scope(options)
        reports = []
        try:
            with embedding_build_database(scope.budget, using="default"):
                space = load_embedding_space(options["space_key"], using="default")
                documents = scope.documents()
                plans = [plan_embedding_reindex(doc, space=space) for doc in documents]
                reports = [report for _, report in plans]
                if not options["dry_run"]:
                    for selection, report in plans:
                        if report["status"] != "planned":
                            continue
                        check_embedding_deadline(scope.budget)
                        report["status"] = "scheduling"
                        report["generation_key"] = rebuild_document_embeddings(
                            selection, space_key=space.space_key, using="default",
                            timeout_ms=options["timeout_ms"],
                        )
                        report["status"] = "queued"
        except (LookupError, ValueError, ObjectDoesNotExist, DatabaseError, TimeoutError) as error:
            if reports:
                self._report(options, scope, reports)
            raise CommandError("Verified embedding scope or bounded rebuild unavailable.") from error
        self._report(options, scope, reports)
        if any(row["status"] not in ("planned", "queued") for row in reports):
            raise CommandError("One or more owners were excluded; inspect the metadata report.")

    def _report(self, options, scope, reports):
        self.stdout.write(json.dumps(dict(
            version="embedding-reindex-plan-v1", dry_run=options["dry_run"],
            space_key=options["space_key"], input_integrity_verified=False,
            dense_coverage_verified=False, documents=reports,
            counts=dict(Counter(row["status"] for row in reports)),
            next_after_pkid=reports[-1]["document_pkid"] if reports and not scope.exact else None,
        )))
