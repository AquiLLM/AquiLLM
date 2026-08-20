from __future__ import annotations

import json

from django.core.management.base import BaseCommand

from apps.knowledge_graph.projection.reconciler import reconcile_graph_projections


class Command(BaseCommand):
    help = "Reconcile bounded PostgreSQL projection authority pages."

    def add_arguments(self, parser):
        parser.add_argument("--page-size", type=int, default=500)
        parser.add_argument("--dry-run", action="store_true")

    def handle(self, *args, **options):
        summary = reconcile_graph_projections(
            page_size=options["page_size"], dry_run=options["dry_run"]
        )
        self.stdout.write(
            json.dumps(
                {
                    "examined_count": summary.examined_count,
                    "enqueued_count": summary.enqueued_count,
                },
                sort_keys=True,
            )
        )
