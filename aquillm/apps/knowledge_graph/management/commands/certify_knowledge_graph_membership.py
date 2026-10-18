"""Explicit bounded maintenance; never schedules or rebuilds graph generations."""

from time import monotonic
from uuid import UUID

from django.core.management.base import BaseCommand, CommandError

from apps.knowledge_graph.models import CollectionGraphProjection
from apps.knowledge_graph.projection.database_deadline import (
    projection_database,
    projection_operation,
)
from apps.knowledge_graph.projection.membership_certification import (
    certify_ready_membership,
)
from apps.knowledge_graph.projection.runtime import load_projection_runtime_settings


class Command(BaseCommand):
    help = (
        "Certify at most 128 current ready membership occurrences within 300 seconds."
    )

    def add_arguments(self, parser):
        parser.add_argument("--limit", type=int, default=32)
        parser.add_argument(
            "--after-projection-id", help="Resume after this canonical UUID."
        )

    def handle(self, *args, **options):
        limit = options["limit"]
        if type(limit) is not int or not 1 <= limit <= 128:
            raise CommandError("limit must be in 1..128")
        raw_after = options.get("after_projection_id")
        after = None
        if raw_after is not None:
            try:
                after = UUID(raw_after) if type(raw_after) is str else None
                if after is None or str(after) != raw_after:
                    raise ValueError
            except (TypeError, ValueError):
                raise CommandError(
                    "after-projection-id must be a canonical UUID"
                ) from None
        settings = load_projection_runtime_settings()
        if not (
            getattr(settings, "membership_certificate_enabled", False)
            or getattr(settings, "membership_certificate_issuance_enabled", False)
        ):
            self.stdout.write("failure_code=certificate_disabled")
            self.stdout.write("examined=0 certified=0 failed=0 not_certified=0")
            return
        certified = failed = examined = not_certified = 0
        last_examined = after
        exhausted = False
        deadline = monotonic() + 300
        try:
            with projection_operation(300000, deadline=deadline, call_timeout_ms=5000):
                with projection_database("projection_source"):
                    filters = {"state": "ready"}
                    if after is not None:
                        filters["pk__gt"] = after
                    identifiers = tuple(
                        CollectionGraphProjection.objects.using("projection_source")
                        .filter(**filters)
                        .order_by("id")
                        .values_list("pk", flat=True)[: limit + 1]
                    )
                for identifier in identifiers[:limit]:
                    if monotonic() >= deadline:
                        break
                    examined += 1
                    last_examined = identifier
                    try:
                        if certify_ready_membership(
                            projection_id=identifier, settings=settings
                        ):
                            certified += 1
                        else:
                            not_certified += 1
                    except TimeoutError:
                        failed += 1
                        self.stdout.write("failure_code=certificate_timeout")
                    except Exception:
                        failed += 1
                        self.stdout.write("failure_code=certificate_failed")
                exhausted = examined == len(identifiers)
        except TimeoutError:
            failed += 1
            self.stdout.write("failure_code=certificate_pass_timeout")
        except Exception:
            failed += 1
            self.stdout.write("failure_code=certificate_pass_failed")
        self.stdout.write(
            f"examined={examined} certified={certified} failed={failed} "
            f"not_certified={not_certified}"
        )
        self.stdout.write(
            f"next_after_projection_id={last_examined or 'none'} "
            f"exhausted={int(exhausted)}"
        )
