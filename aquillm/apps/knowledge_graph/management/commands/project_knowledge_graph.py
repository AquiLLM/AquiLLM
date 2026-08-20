from __future__ import annotations

import json

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from apps.knowledge_graph.models import GraphArtifact
from apps.knowledge_graph.projection.lifecycle import (
    enqueue_collection_projection_locked,
)
from apps.knowledge_graph.projection.reconciler import reconcile_graph_projections
from apps.knowledge_graph.projection.runtime import load_projection_runtime_settings


class Command(BaseCommand):
    help = "Enqueue current collection graph projections without exposing graph data."

    def add_arguments(self, parser):
        settings = load_projection_runtime_settings()
        parser.add_argument("--collection", type=int)
        parser.add_argument("--all", action="store_true")
        parser.add_argument("--dry-run", action="store_true")
        parser.add_argument(
            "--page-size", type=int, default=settings.projection_batch_size
        )

    def handle(self, *args, **options):
        collection_id, all_collections = options["collection"], options["all"]
        if (collection_id is None) == (all_collections is False):
            raise CommandError("choose exactly one of --collection or --all")
        if collection_id is not None and collection_id < 1:
            raise CommandError("--collection must be positive")
        if all_collections:
            summary = reconcile_graph_projections(
                page_size=options["page_size"], dry_run=options["dry_run"]
            )
            payload = {
                "examined_count": summary.examined_count,
                "enqueued_count": summary.enqueued_count,
            }
        else:
            artifact_id = (
                GraphArtifact.objects.filter(
                    collection_scope_id=collection_id,
                    scope_type="collection",
                    status="active",
                    evaluation_only=False,
                )
                .values_list("pk", flat=True)
                .first()
            )
            if artifact_id is None:
                raise CommandError("collection has no active graph artifact")
            if not options["dry_run"]:
                with transaction.atomic():
                    enqueue_collection_projection_locked(
                        collection_id=collection_id,
                        artifact_id=artifact_id,
                        using="default",
                    )
            payload = {
                "examined_count": 1,
                "enqueued_count": 0 if options["dry_run"] else 1,
            }
        self.stdout.write(json.dumps(payload, sort_keys=True))
