"""Explicit bounded maintenance; never schedules or rebuilds graph generations."""

from time import monotonic

from django.core.management.base import BaseCommand, CommandError

from apps.knowledge_graph.models import CollectionGraphProjection
from apps.knowledge_graph.projection.database_deadline import projection_database, projection_operation
from apps.knowledge_graph.projection.membership_certification import certify_ready_membership


class Command(BaseCommand):
    help = "Certify at most 128 current ready membership occurrences within 300 seconds."

    def add_arguments(self, parser):
        parser.add_argument("--limit", type=int, default=32)

    def handle(self, *args, **options):
        limit = options["limit"]
        if type(limit) is not int or not 1 <= limit <= 128:
            raise CommandError("limit must be in 1..128")
        certified = failed = examined = 0
        deadline = monotonic() + 300
        try:
            with projection_operation(300000, deadline=deadline, call_timeout_ms=5000):
                with projection_database("projection_source"):
                    identifiers = tuple(CollectionGraphProjection.objects.using("projection_source")
                        .filter(state="ready").order_by("id").values_list("pk", flat=True)[:limit])
                for identifier in identifiers:
                    if monotonic() >= deadline:
                        break
                    examined += 1
                    try:
                        certified += bool(certify_ready_membership(projection_id=identifier))
                    except TimeoutError:
                        failed += 1
                        self.stdout.write("failure_code=certificate_timeout")
                    except Exception:
                        failed += 1
                        self.stdout.write("failure_code=certificate_failed")
        except TimeoutError:
            failed += 1
            self.stdout.write("failure_code=certificate_pass_timeout")
        except Exception:
            failed += 1
            self.stdout.write("failure_code=certificate_pass_failed")
        self.stdout.write(f"examined={examined} certified={certified} failed={failed}")
