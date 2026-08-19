from __future__ import annotations

import uuid

from celery import current_app
from django.core.management.base import BaseCommand, CommandError

from apps.knowledge_graph.services.builds import (
    RebuildPublicationError,
    assert_evaluation_bypass,
    create_rebuild_request,
    preview_rebuild,
)
from lib.knowledge_graph.config import get_build_enabled

_EXTRACTION_QUEUE = "knowledge-graph-extraction"


def _extraction_worker_available() -> bool:
    """Check queue capability without importing or initializing extractor ML."""

    try:
        replies = current_app.control.inspect(timeout=1.0).active_queues()
    except Exception:
        return False
    if type(replies) is not dict:
        return False
    return any(
        type(queue) is dict and queue.get("name") == _EXTRACTION_QUEUE
        for queues in replies.values()
        if type(queues) is list
        for queue in queues
    )


def _positive_collection_id(value: str) -> int:
    try:
        collection_id = int(value, 10)
    except (TypeError, ValueError) as exc:
        raise ValueError("collection must be a positive integer") from exc
    if collection_id <= 0 or collection_id >= 2**63 or str(collection_id) != value:
        raise ValueError("collection must be a positive integer")
    return collection_id


class Command(BaseCommand):
    help = "Queue a durable knowledge-graph rebuild request."

    def add_arguments(self, parser) -> None:
        scope = parser.add_mutually_exclusive_group(required=True)
        scope.add_argument("--document", type=uuid.UUID)
        scope.add_argument("--collection", type=_positive_collection_id)
        scope.add_argument("--all", action="store_true", dest="all_scopes")
        parser.add_argument("--request-id", type=uuid.UUID)
        parser.add_argument("--dry-run", action="store_true")
        parser.add_argument("--yes", action="store_true")
        parser.add_argument("--eval-only", action="store_true")

    def handle(self, *args, **options):
        document_id = options["document"]
        collection_id = options["collection"]
        all_scopes = options["all_scopes"]
        dry_run = options["dry_run"]
        evaluation_only = options["eval_only"]

        if all_scopes and not dry_run and not options["yes"]:
            raise CommandError("operator-wide --all rebuild requires --yes")
        if evaluation_only and collection_id is None:
            raise CommandError("--eval-only requires one concrete --collection")
        try:
            assert_evaluation_bypass(evaluation_only)
        except (PermissionError, ValueError) as exc:
            raise CommandError(str(exc)) from exc

        if document_id is not None:
            scope_type, scope_id = "document", document_id
        elif collection_id is not None:
            scope_type, scope_id = "collection", collection_id
        elif all_scopes:
            scope_type, scope_id = "all", None
        else:  # Defensive: argparse's required mutually-exclusive group owns this.
            raise CommandError("exactly one rebuild scope is required")

        if dry_run:
            report = preview_rebuild(scope_type=scope_type, scope_id=scope_id)
            document_count = int(report.get("document_count", 0))
            collection_count = int(report.get("collection_count", 0))
            self.stdout.write(
                f"dry-run documents={document_count} collections={collection_count}"
            )
            return

        if not evaluation_only:
            try:
                build_enabled = get_build_enabled()
            except ValueError as exc:
                raise CommandError(
                    "knowledge-graph build runtime is misconfigured"
                ) from exc
            if not build_enabled:
                raise CommandError("knowledge-graph build worker is disabled")
        if not _extraction_worker_available():
            raise CommandError("knowledge-graph extraction worker is unavailable")

        request_id = options["request_id"] or uuid.uuid4()
        try:
            request = create_rebuild_request(
                scope_type=scope_type,
                scope_id=scope_id,
                request_id=request_id,
                evaluation_only=evaluation_only,
            )
        except RebuildPublicationError as exc:
            self.stdout.write(str(exc.request_id))
            raise CommandError(f"rebuild publication failed: {exc.error_code}") from exc
        except (LookupError, PermissionError, RuntimeError, ValueError) as exc:
            raise CommandError(str(exc)) from exc
        self.stdout.write(str(request.pk))
