from __future__ import annotations

import re
import secrets
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from apps.knowledge_graph.services.ontology import (
    OntologyValidationError,
    activate_ontology,
    load_ontology,
)

_ONTOLOGY_ROOT = Path(__file__).resolve().parents[2] / "ontologies"
_CHECKSUM_PATTERN = re.compile(r"[0-9a-f]{64}")


def _resolve_ontology_path(value: str) -> Path:
    if type(value) is not str or not value or len(value) > 256 or "\x00" in value:
        raise CommandError("ontology path is invalid")
    relative_path = Path(value)
    if relative_path.is_absolute() or relative_path.suffix != ".yaml":
        raise CommandError("ontology path must be inside the ontology directory")

    try:
        root = _ONTOLOGY_ROOT.resolve(strict=True)
        candidate = (root / relative_path).resolve(strict=True)
        candidate.relative_to(root)
    except (OSError, RuntimeError, ValueError) as error:
        raise CommandError(
            "ontology path must be inside the ontology directory"
        ) from error
    if not candidate.is_file():
        raise CommandError("ontology path must name a regular YAML file")
    return candidate


class Command(BaseCommand):
    help = "Validate or explicitly activate a checked-in knowledge-graph ontology."

    def add_arguments(self, parser) -> None:
        parser.add_argument("--path", required=True)
        parser.add_argument("--expected-checksum", required=True)
        parser.add_argument("--dry-run", action="store_true")
        parser.add_argument("--yes", action="store_true")

    def handle(self, *args, **options) -> None:
        expected_checksum = options["expected_checksum"]
        if (
            type(expected_checksum) is not str
            or _CHECKSUM_PATTERN.fullmatch(expected_checksum) is None
        ):
            raise CommandError("expected checksum must be lowercase SHA-256")
        ontology_path = _resolve_ontology_path(options["path"])
        if not options["dry_run"] and not options["yes"]:
            raise CommandError("activation requires explicit --yes confirmation")

        try:
            definition = load_ontology(ontology_path)
        except (OSError, UnicodeError, OntologyValidationError) as error:
            raise CommandError("ontology validation failed") from error
        if not secrets.compare_digest(definition.checksum, expected_checksum):
            raise CommandError("ontology checksum does not match expected checksum")

        if not options["dry_run"]:
            try:
                activate_ontology(definition)
            except OntologyValidationError as error:
                raise CommandError("ontology activation failed") from error
        self.stdout.write(
            f"version={definition.version} checksum={definition.checksum}"
        )
