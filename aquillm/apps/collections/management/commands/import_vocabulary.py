"""Bounded authenticated import; publishing is explicit and never repins."""
import json
from getpass import getpass
from pathlib import Path

from django.contrib.auth import authenticate
from django.core.management.base import BaseCommand, CommandError

from apps.collections.models import VocabularySet
from apps.collections.services.vocabulary import publish_vocabulary, save_draft
from apps.collections.services.vocabulary_contracts import MAX_BYTES


class Command(BaseCommand):
    help = "Import a bounded draft using current MANAGE permission and revision CAS."

    def add_arguments(self, parser):
        parser.add_argument("path")
        parser.add_argument("--username", required=True)
        parser.add_argument("--vocabulary", type=int, required=True)
        parser.add_argument("--expected-revision", type=int, required=True)
        parser.add_argument("--publish", action="store_true")

    def handle(self, *args, **options):
        actor = authenticate(username=options["username"], password=getpass("Password: "))
        if actor is None or not actor.is_active or not actor.is_staff:
            raise CommandError("Current authenticated staff actor required.")
        try:
            with Path(options["path"]).open("rb") as stream:
                raw = stream.read(MAX_BYTES + 1)
            if len(raw) > MAX_BYTES:
                raise ValueError("vocabulary byte limit")
            entries = json.loads(raw)
            vocabulary = VocabularySet.objects.get(pk=options["vocabulary"])
            draft = save_draft(actor=actor, vocabulary=vocabulary,
                expected_revision=options["expected_revision"], entries=entries)
            if options["publish"]:
                version = publish_vocabulary(actor=actor, vocabulary=vocabulary,
                    expected_revision=draft.revision)
                self.stdout.write(version.version_key)
            else:
                self.stdout.write(f"Draft revision {draft.revision}")
        except (OSError, ValueError, PermissionError, VocabularySet.DoesNotExist) as exc:
            raise CommandError(str(exc)) from exc
