from __future__ import annotations

import json
import uuid

from django.core.management.base import BaseCommand, CommandError

from apps.knowledge_graph.services.inspection import inspect_graph_state


def _positive_collection_id(value: str) -> int:
    try:
        result = int(value, 10)
    except (TypeError, ValueError) as exc:
        raise ValueError("collection must be a positive integer") from exc
    if result <= 0 or result >= 2**63 or str(result) != value:
        raise ValueError("collection must be a positive integer")
    return result


def _bounded_timeout(value: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("timeout must be a number") from exc
    if not 0 < result <= 3_600:
        raise ValueError("timeout must be in (0, 3600]")
    return result


class Command(BaseCommand):
    help = "Inspect bounded, privacy-safe knowledge-graph lifecycle state."

    def add_arguments(self, parser) -> None:
        parser.add_argument("--document", type=uuid.UUID)
        parser.add_argument("--collection", type=_positive_collection_id)
        parser.add_argument("--request-id", type=uuid.UUID)
        parser.add_argument("--wait", action="store_true")
        parser.add_argument("--timeout-seconds", type=_bounded_timeout, default=60.0)

    def handle(self, *args, **options):
        if options["wait"] and options["request_id"] is None:
            raise CommandError("--wait requires --request-id")
        try:
            report = inspect_graph_state(
                document_id=options["document"],
                collection_id=options["collection"],
                request_id=options["request_id"],
                wait=options["wait"],
                timeout_seconds=options["timeout_seconds"],
            )
        except (LookupError, RuntimeError, TimeoutError, ValueError) as exc:
            raise CommandError(str(exc)) from exc
        self.stdout.write(json.dumps(report, sort_keys=True, separators=(",", ":")))
