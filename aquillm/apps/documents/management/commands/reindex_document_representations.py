"""Explicit bounded operators over immutable extraction revisions; no parser I/O."""
import json
from uuid import UUID

from django.core.management.base import BaseCommand, CommandError

from apps.documents.models import DESCENDED_FROM_DOCUMENT
from apps.documents.services.representation_reindex import (
    execute_plan, plan_document, registered_chunker_keys,
)


class Command(BaseCommand):
    help = ("Build structural generations from exact persisted extraction revisions, or a nonledger "
            "RawTextDocument canonical snapshot. Dry-run is metadata/policy planning only; it does "
            "not certify byte integrity or build success. No parser, provider, or enqueue is used.")

    def add_arguments(self, parser):
        parser.add_argument("--database", default="default")
        parser.add_argument("--model-label")
        parser.add_argument("--document-pkid", type=int)
        parser.add_argument("--document-id")
        parser.add_argument("--collection-id", type=int)
        parser.add_argument("--limit", type=int)
        parser.add_argument("--after-pkid", type=int)
        parser.add_argument("--parser-key")
        parser.add_argument("--chunker-key")
        parser.add_argument("--extraction-revision-id", type=int)
        parser.add_argument("--dry-run", action="store_true")
        parser.add_argument("--show-policies", action="store_true")

    def handle(self, *args, **options):
        if options["database"] != "default":
            raise CommandError("Only the default database is supported.")
        keys = registered_chunker_keys()
        if options["show_policies"]:
            self.stdout.write(json.dumps(dict(chunker_keys=keys,
                canonical_parser_key="canonical-snapshot-v1",
                extraction_parser_key="Use the exact persisted revision parser_key; aliases are unsupported.")))
            return
        models = {model._meta.label_lower: model for model in DESCENDED_FROM_DOCUMENT}
        model = models.get(options["model_label"])
        parser_key = options["parser_key"]
        if model is None:
            raise CommandError("An exact allowlisted concrete model label is required.")
        if not isinstance(parser_key, str) or not parser_key or len(parser_key) > 200:
            raise CommandError("An exact persisted parser key is required.")
        if options["chunker_key"] not in keys:
            raise CommandError("Unknown chunker policy. Use --show-policies for configured identities.")
        revision = options["extraction_revision_id"]
        if revision is not None and (type(revision) is not int or revision <= 0):
            raise CommandError("Extraction revision ID must be positive.")
        exact = options["document_pkid"] is not None or options["document_id"] is not None
        query = model._base_manager.using("default").only("pkid", "id", "collection_id", "full_text_hash")
        if exact:
            if any(options[key] is not None for key in ("collection_id", "limit", "after_pkid")):
                raise CommandError("Exact owner and inventory options are mutually exclusive.")
            try:
                identifier = UUID(options["document_id"])
                pkid = options["document_pkid"]
                if type(pkid) is not int or pkid <= 0:
                    raise ValueError
            except (TypeError, ValueError, AttributeError):
                raise CommandError("Exact owner requires positive --document-pkid and valid --document-id.")
            documents = list(query.filter(pkid=pkid, id=identifier)[:1])
            if not documents:
                raise CommandError("Exact owner does not exist.")
        else:
            collection, limit, after = (options[key] for key in ("collection_id", "limit", "after_pkid"))
            if (type(collection) is not int or collection <= 0 or type(limit) is not int
                or not 1 <= limit <= 1000 or type(after) is not int or after < 0):
                raise CommandError("Inventory requires positive --collection-id, --limit 1..1000, and --after-pkid >=0.")
            if revision is not None:
                raise CommandError("An explicit extraction revision requires an exact owner.")
            documents = list(query.filter(collection_id=collection, pkid__gt=after).order_by("pkid")[:limit])
        plans = [plan_document(doc, parser_key=parser_key, chunker_key=options["chunker_key"],
                               revision_id=revision, using="default") for doc in documents]
        reports = [plan.report(using="default") for plan in plans]
        if not options["dry_run"]:
            for plan, report in zip(plans, reports, strict=True):
                if plan.status == "planned":
                    report["status"] = execute_plan(plan, using="default")
        self.stdout.write(json.dumps(dict(dry_run=options["dry_run"], integrity_verified=False,
            documents=reports, next_after_pkid=documents[-1].pkid if documents and not exact else None)))
        if any(report["status"] not in ("planned", "activated", "already_active") for report in reports):
            raise CommandError("One or more owners were excluded or did not activate; inspect the metadata report.")
