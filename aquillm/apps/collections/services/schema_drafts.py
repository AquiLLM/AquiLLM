"""Transactional editing and generation writes for shared schema drafts."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from django.db import transaction
from django.utils import timezone

from apps.collections.models import Collection, CollectionSchemaDraft

from . import schema as core
from .schema import (
    SchemaGenerationDraftConflict,
    SchemaOperationError,
    SchemaRevisionConflict,
    _capabilities,
    _current_version,
    _published_definitions,
    canonicalize_definitions,
)


def create_draft(collection: Collection, user) -> CollectionSchemaDraft:
    with transaction.atomic():
        locked_collection = core.Collection.objects.select_for_update().get(
            pk=collection.pk
        )
        existing = (
            core.CollectionSchemaDraft.objects.select_for_update()
            .filter(collection=collection)
            .first()
        )
        if existing is not None:
            return existing
        base = _current_version(locked_collection)
        definitions = (
            _published_definitions(base.definitions)
            if base is not None
            else {"entities": [], "relations": []}
        )
        return core.CollectionSchemaDraft.objects.create(
            collection=collection,
            base_version=base,
            definitions=definitions,
            last_editor=user,
        )


def _conflict_fields(kind: str, key: str, draft, attempted_values) -> list[dict]:
    group_name = "entities" if kind == "entity" else "relations"
    rows = canonicalize_definitions(draft.definitions)[group_name]
    server = next((row.get("values", {}) for row in rows if row["key"] == key), {})
    attempted = attempted_values if type(attempted_values) is dict else {}
    fields = [
        {
            "field": field,
            "server_value": server.get(field),
            "attempted_value": value,
        }
        for field, value in attempted.items()
        if server.get(field) != value
    ]
    return [{"kind": kind, "key": key, "fields": fields}]


def _locked_draft(
    collection: Collection, revision: int | None, *, kind=None, key=None, attempted=None
):
    draft = (
        core.CollectionSchemaDraft.objects.select_for_update()
        .filter(collection=collection)
        .first()
    )
    if draft is None:
        raise SchemaOperationError("draft_not_found", status=404)
    if revision is None or revision != draft.revision:
        definitions = (
            _conflict_fields(kind, key, draft, attempted)
            if kind is not None and key is not None
            else []
        )
        raise SchemaRevisionConflict(revision, draft, definitions)
    return draft


def mutate_definition(
    collection: Collection,
    user,
    kind: str,
    key: str,
    revision: int | None,
    values: dict[str, Any] | None,
    *,
    draft_id: str,
) -> CollectionSchemaDraft:
    if kind not in {"entity", "relation"}:
        raise ValueError("unsupported schema definition kind")
    if type(key) is not str or not key:
        raise SchemaOperationError("invalid_definition_key")
    attempted = values or {}
    if values is not None and type(values) is not dict:
        raise SchemaOperationError("invalid_definition")
    with transaction.atomic():
        core.Collection.objects.select_for_update().get(pk=collection.pk)
        draft = _locked_draft(
            collection,
            revision,
            kind=kind,
            key=key,
            attempted=attempted,
        )
        if str(draft.pk) != str(draft_id):
            raise SchemaRevisionConflict(
                revision, draft, _conflict_fields(kind, key, draft, attempted)
            )
        definitions = canonicalize_definitions(draft.definitions)
        group = definitions["entities" if kind == "entity" else "relations"]
        existing = next((row for row in group if row["key"] == key), None)
        if values is None:
            if existing is not None:
                group.remove(existing)
        else:
            normalized_values = deepcopy(values)
            normalized_values["name"] = key
            row = {
                "key": key,
                "origin": existing.get("origin", "collection")
                if existing
                else "collection",
                "change_state": "changed" if existing else "added",
                "capabilities": (
                    deepcopy(existing.get("capabilities"))
                    if existing and existing.get("capabilities")
                    else _capabilities(kind)
                ),
                "values": normalized_values,
            }
            if existing is None:
                group.append(row)
            else:
                group[group.index(existing)] = row
        draft.definitions = canonicalize_definitions(definitions)
        draft.revision += 1
        draft.last_editor = user
        draft.save(
            update_fields=("definitions", "revision", "last_editor", "updated_at")
        )
        return draft


def discard_draft(collection: Collection, draft_id, revision: int | None) -> None:
    with transaction.atomic():
        core.Collection.objects.select_for_update().get(pk=collection.pk)
        draft = _locked_draft(collection, revision)
        if str(draft.pk) != str(draft_id):
            raise SchemaOperationError("draft_identity_mismatch", status=409)
        draft.delete()


def write_generated_draft(run_id, definitions, statistics):
    from django.db import transaction

    from apps.collections.models import (
        CollectionSchemaGenerationRun,
    )

    canonical = canonicalize_definitions(definitions)
    collection_id = CollectionSchemaGenerationRun.objects.values_list(
        "collection_id", flat=True
    ).get(pk=run_id)
    with transaction.atomic():
        core.Collection.objects.select_for_update().get(pk=collection_id)
        run = (
            CollectionSchemaGenerationRun.objects.select_for_update()
            .select_related("collection")
            .get(pk=run_id)
        )
        if run.collection_id != collection_id:
            raise SchemaGenerationDraftConflict("run_collection_changed")
        if run.status != CollectionSchemaGenerationRun.Status.RUNNING:
            raise ValueError("schema generation run must be running")
        current = (
            core.CollectionSchemaDraft.objects.select_for_update()
            .filter(collection=run.collection)
            .first()
        )
        current_revision = current.revision if current is not None else None
        current_id = current.pk if current is not None else None
        if (
            current_id != run.base_draft_id
            or current_revision != run.base_draft_revision
        ):
            raise SchemaGenerationDraftConflict("draft_conflict")
        if current is None:
            if run.requested_by is None:
                raise ValueError("schema generation run requester is unavailable")
            draft = core.CollectionSchemaDraft.objects.create(
                collection=run.collection,
                base_version=_current_version(run.collection),
                definitions=canonical,
                last_editor=run.requested_by,
            )
        else:
            current.definitions = canonical
            current.revision += 1
            update_fields = ["definitions", "revision", "updated_at"]
            if run.requested_by is not None:
                current.last_editor = run.requested_by
                update_fields.append("last_editor")
            current.save(update_fields=tuple(update_fields))
            draft = current
        run.statistics = deepcopy(statistics)
        run.status = CollectionSchemaGenerationRun.Status.SUCCEEDED
        run.error_code = ""
        run.completed_at = timezone.now()
        run.save(
            update_fields=(
                "statistics",
                "status",
                "error_code",
                "completed_at",
                "updated_at",
            )
        )
        return draft
