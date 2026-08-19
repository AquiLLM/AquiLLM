from __future__ import annotations

import json

from django.core.management.base import BaseCommand, CommandError

from apps.knowledge_graph.services.pruning import prune_graph_artifacts


class Command(BaseCommand):
    help = "Preview or execute bounded knowledge-graph artifact pruning."

    def add_arguments(self, parser) -> None:
        parser.add_argument("--execute", action="store_true")
        parser.add_argument("--batch-size", type=int)

    def handle(self, *args, **options):
        kwargs: dict[str, object] = {"execute": bool(options["execute"])}
        if options["batch_size"] is not None:
            if not 1 <= options["batch_size"] <= 1_000:
                raise CommandError("--batch-size must be in [1, 1000]")
            kwargs["batch_size"] = options["batch_size"]
        try:
            report = prune_graph_artifacts(**kwargs)
        except (RuntimeError, ValueError) as exc:
            raise CommandError(str(exc)) from exc
        payload = report.as_dict() if hasattr(report, "as_dict") else report
        self.stdout.write(json.dumps(payload, sort_keys=True, separators=(",", ":")))
