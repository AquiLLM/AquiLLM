from __future__ import annotations

import json

from django.core.management.base import BaseCommand, CommandError

from apps.knowledge_graph.projection.reconciler import inspect_projection_authority
from apps.knowledge_graph.projection.runtime import load_projection_runtime_settings


class Command(BaseCommand):
    help = "Inspect bounded opaque projection authority counts."

    def add_arguments(self, parser):
        settings = load_projection_runtime_settings()
        parser.add_argument("--collection", type=int)
        parser.add_argument("--all", action="store_true")
        parser.add_argument(
            "--page-size", type=int, default=settings.projection_batch_size
        )

    def handle(self, *args, **options):
        collection_id, all_collections = options["collection"], options["all"]
        if (collection_id is None) == (all_collections is False):
            raise CommandError("choose exactly one of --collection or --all")
        if collection_id is not None and collection_id < 1:
            raise CommandError("--collection must be positive")
        payload = inspect_projection_authority(
            collection_id=collection_id,
            all_collections=all_collections,
            page_size=options["page_size"],
        )
        self.stdout.write(json.dumps(payload, sort_keys=True))
