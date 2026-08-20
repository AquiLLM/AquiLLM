from __future__ import annotations

import json

from django.core.management.base import BaseCommand, CommandError

from apps.knowledge_graph.projection.reconciler import inspect_projection_authority


class Command(BaseCommand):
    help = "Inspect bounded opaque projection authority counts."

    def add_arguments(self, parser):
        parser.add_argument("--collection", type=int)
        parser.add_argument("--all", action="store_true")
        parser.add_argument("--page-size", type=int, default=500)

    def handle(self, *args, **options):
        collection_id, all_collections = options["collection"], options["all"]
        if (collection_id is None) == (all_collections is False):
            raise CommandError("choose exactly one of --collection or --all")
        payload = inspect_projection_authority(
            collection_id=collection_id,
            all_collections=all_collections,
            page_size=options["page_size"],
        )
        self.stdout.write(json.dumps(payload, sort_keys=True))
