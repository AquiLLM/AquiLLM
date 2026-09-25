from __future__ import annotations

from django.core.management.base import BaseCommand, CommandError

from apps.knowledge_graph.evals.fixture_seed import (
    FIXTURE_ID,
    FixtureSeedError,
    FixtureSeedResult,
    cleanup_fixture,
    seed_fixture,
)


def _write_result(
    command: BaseCommand, result: FixtureSeedResult, *, cleanup: bool
) -> None:
    command.stdout.write(
        " ".join(
            (
                f"fixture_id={FIXTURE_ID}",
                f"fixture_checksum={result.fixture_checksum}",
                f"manifest_checksum={result.manifest_checksum}",
                f"manifest_path={result.manifest_path}",
            )
        )
    )
    command.stdout.write(
        " ".join(
            (
                f"{'collections_deleted' if cleanup else 'collections'}="
                f"{result.collection_count}",
                f"documents={result.document_count}",
                f"chunks={result.chunk_count}",
            )
        )
    )
    for collection_id, request_id in result.authorized_scope:
        command.stdout.write(
            f"collection_id={collection_id} rebuild_request_id={request_id}"
        )


class Command(BaseCommand):
    help = "Create or clean the deterministic synthetic knowledge-graph eval fixture."

    def add_arguments(self, parser) -> None:
        parser.add_argument("--fixture-manifest", required=True)
        parser.add_argument("--cleanup", action="store_true")
        parser.add_argument("--expected-manifest-checksum")

    def handle(self, *args, **options) -> None:
        cleanup = options["cleanup"]
        expected = options["expected_manifest_checksum"]
        if cleanup and expected is None:
            raise CommandError("cleanup requires --expected-manifest-checksum")
        if not cleanup and expected is not None:
            raise CommandError(
                "--expected-manifest-checksum is valid only with --cleanup"
            )
        try:
            result = (
                cleanup_fixture(
                    options["fixture_manifest"],
                    expected_manifest_checksum=expected,
                )
                if cleanup
                else seed_fixture(options["fixture_manifest"])
            )
        except FixtureSeedError as error:
            raise CommandError(str(error)) from error
        _write_result(self, result, cleanup=cleanup)
