from __future__ import annotations

import json

from django.core.management.base import BaseCommand, CommandError

from apps.knowledge_graph.projection.reconciler import reconcile_graph_projections
from apps.knowledge_graph.projection.runtime import load_projection_runtime_settings


class Command(BaseCommand):
    help = "Reconcile bounded PostgreSQL projection authority pages."

    def add_arguments(self, parser):
        settings = load_projection_runtime_settings()
        parser.add_argument("--collection", type=int)
        parser.add_argument("--all", action="store_true")
        parser.add_argument(
            "--page-size", type=int, default=settings.projection_batch_size
        )
        parser.add_argument("--dry-run", action="store_true")

    def handle(self, *args, **options):
        collection_id, all_collections = options["collection"], options["all"]
        if (collection_id is None) == (all_collections is False):
            raise CommandError("choose exactly one of --collection or --all")
        if collection_id is not None and collection_id < 1:
            raise CommandError("--collection must be positive")
        summary = reconcile_graph_projections(
            page_size=options["page_size"],
            dry_run=options["dry_run"],
            collection_id=collection_id,
        )
        self.stdout.write(
            json.dumps(
                {
                    "examined_count": summary.examined_count,
                    "enqueued_count": summary.enqueued_count,
                    "drift_count": summary.drift_count,
                    "orphan_count": summary.orphan_count,
                    "replayed_count": summary.replayed_count,
                },
                sort_keys=True,
            )
        )
