from __future__ import annotations

import json
from uuid import UUID

from django.core.management.base import BaseCommand, CommandError

from apps.knowledge_graph.projection.reconciler import (
    prune_graph_projection_generations,
)
from apps.knowledge_graph.projection.runtime import load_projection_runtime_settings


class Command(BaseCommand):
    help = "Prune bounded superseded Memgraph projection generations."

    def add_arguments(self, parser):
        settings = load_projection_runtime_settings()
        parser.add_argument("--projection")
        parser.add_argument("--collection", type=int)
        parser.add_argument("--all", action="store_true")
        parser.add_argument(
            "--page-size", type=int, default=settings.projection_batch_size
        )
        parser.add_argument("--retain", type=int, default=settings.projection_retention)
        parser.add_argument("--dry-run", action="store_true")

    def handle(self, *args, **options):
        projection_value = options["projection"]
        collection_id = options["collection"]
        all_collections = options["all"]
        selectors = (
            projection_value is not None,
            collection_id is not None,
            all_collections,
        )
        if sum(selectors) != 1:
            raise CommandError(
                "choose exactly one of --projection, --collection, or --all"
            )
        if collection_id is not None and collection_id < 1:
            raise CommandError("--collection must be positive")
        projection_id = None
        if projection_value is not None:
            try:
                projection_id = UUID(projection_value)
            except (AttributeError, TypeError, ValueError) as exc:
                raise CommandError("--projection must be a canonical UUID") from exc
            if str(projection_id) != projection_value:
                raise CommandError("--projection must be a canonical UUID")
        summary = prune_graph_projection_generations(
            page_size=options["page_size"],
            retain=options["retain"],
            dry_run=options["dry_run"],
            projection_id=projection_id,
            collection_id=collection_id,
        )
        self.stdout.write(
            json.dumps(
                {
                    "candidate_count": summary.candidate_count,
                    "deleted_count": summary.deleted_count,
                },
                sort_keys=True,
            )
        )
