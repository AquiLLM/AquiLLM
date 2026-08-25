from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from apps.collections.models import (
    Collection,
    CollectionPermission,
    CollectionSchemaDraft,
    CollectionSchemaGenerationRun,
)
from apps.collections.services.schema import canonicalize_definitions
from apps.collections.services.schema_generation import (
    _locked_collection_source_signature,
    collection_has_eligible_text,
)
from apps.collections.tasks.schema_generation import enqueue_schema_generation


def _requester_for(collection, draft):
    if draft is not None:
        return draft.last_editor
    permission_scope = collection
    while permission_scope is not None:
        permission = (
            CollectionPermission.objects.filter(
                collection=permission_scope,
                permission__in=("EDIT", "MANAGE"),
            )
            .select_related("user")
            .order_by("id")
            .first()
        )
        if permission is not None:
            return permission.user
        permission_scope = permission_scope.parent
    return None


class Command(BaseCommand):
    help = "Queue schema generation for collections without a published schema."

    def add_arguments(self, parser):
        scope = parser.add_mutually_exclusive_group(required=True)
        scope.add_argument("--collection", type=int)
        scope.add_argument("--all", action="store_true", dest="all_collections")
        parser.add_argument("--yes", action="store_true")

    def handle(self, *args, **options):
        collection_id = options["collection"]
        if options["all_collections"] and not options["yes"]:
            raise CommandError("operator-wide schema generation requires --yes")
        if collection_id is not None and collection_id <= 0:
            raise CommandError("collection must be a positive integer")

        collection_ids = (
            Collection.objects.filter(
                current_schema_version__isnull=True,
                **({"pk": collection_id} if collection_id is not None else {}),
            )
            .order_by("pk")
            .values_list("pk", flat=True)
        )
        queued = reused = skipped = 0
        for candidate_id in tuple(collection_ids):
            run_id = None
            with transaction.atomic():
                collection = Collection.objects.select_for_update().get(pk=candidate_id)
                if collection.current_schema_version_id is not None:
                    skipped += 1
                    continue
                draft = (
                    CollectionSchemaDraft.objects.select_for_update()
                    .select_related("last_editor")
                    .filter(collection=collection)
                    .first()
                )
                definitions = (
                    canonicalize_definitions(draft.definitions)
                    if draft is not None
                    else {"entities": [], "relations": []}
                )
                requester = _requester_for(collection, draft)
                if any(definitions.values()) or requester is None:
                    skipped += 1
                    continue
                if not collection_has_eligible_text(collection.pk):
                    skipped += 1
                    continue
                source_signature = _locked_collection_source_signature(collection.pk)
                base_draft_id = draft.pk if draft is not None else None
                base_draft_revision = draft.revision if draft is not None else None
                run = (
                    CollectionSchemaGenerationRun.objects.select_for_update()
                    .filter(
                        collection=collection,
                        status__in=(
                            CollectionSchemaGenerationRun.Status.QUEUED,
                            CollectionSchemaGenerationRun.Status.RUNNING,
                        ),
                    )
                    .first()
                )
                if run is not None:
                    if (
                        draft is not None
                        and run.base_draft_id is None
                        and run.base_draft_revision is None
                    ):
                        run.base_draft_id = base_draft_id
                        run.base_draft_revision = base_draft_revision
                        run.save(
                            update_fields=(
                                "base_draft_id",
                                "base_draft_revision",
                                "updated_at",
                            )
                        )
                    if (
                        run.source_signature != source_signature
                        or run.base_draft_id != base_draft_id
                        or run.base_draft_revision != base_draft_revision
                    ):
                        skipped += 1
                        continue
                    reused += 1
                else:
                    run = CollectionSchemaGenerationRun.objects.create(
                        collection=collection,
                        requested_by=requester,
                        source_signature=source_signature,
                        base_draft_id=base_draft_id,
                        base_draft_revision=base_draft_revision,
                    )
                    queued += 1
                run_id = str(run.pk)
            if run_id is not None:
                enqueue_schema_generation(run_id)

        self.stdout.write(f"queued={queued} reused={reused} skipped={skipped}")
