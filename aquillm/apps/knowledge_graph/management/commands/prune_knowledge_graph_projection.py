from __future__ import annotations

import json

from django.core.management.base import BaseCommand

from apps.knowledge_graph.projection.reconciler import (
    prune_graph_projection_generations,
)


class Command(BaseCommand):
    help = "Prune bounded superseded Memgraph projection generations."

    def add_arguments(self, parser):
        parser.add_argument("--page-size", type=int, default=500)
        parser.add_argument("--retain", type=int, default=2)
        parser.add_argument("--dry-run", action="store_true")

    def handle(self, *args, **options):
        summary = prune_graph_projection_generations(
            page_size=options["page_size"],
            retain=options["retain"],
            dry_run=options["dry_run"],
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
