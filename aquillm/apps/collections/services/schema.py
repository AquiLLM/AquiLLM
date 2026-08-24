from __future__ import annotations

import json
from copy import deepcopy
from hashlib import sha256
from typing import Any

import structlog
import yaml
from django.db import transaction
from django.utils import timezone

from apps.collections.models import (
    Collection,
    CollectionSchemaDraft,
    CollectionSchemaVersion,
)

logger = structlog.stdlib.get_logger(__name__)

CONSTRAINTS = {
    "entity_fields": {
        "name": {"required": True, "max_length": 64},
        "description": {"max_length": 512},
        "default_retrieval_weight": {"min": 0, "max": 1},
        "default_suppression_threshold": {"min": 0, "max": 1},
    },
    "relation_fields": {
        "name": {"required": True, "max_length": 64},
        "direction": {"allowed_values": ["directed", "undirected"]},
    },
}


class SchemaGenerationDraftConflict(RuntimeError):
    """The shared draft no longer matches a generation run's base revision."""


class SchemaRevisionConflict(RuntimeError):
    def __init__(self, attempted: int | None, draft, definitions=None):
        self.attempted = attempted
        self.draft = draft
        self.definitions = definitions or []
        super().__init__("schema draft revision conflict")


class SchemaOperationError(ValueError):
    def __init__(self, code: str, *, status: int = 400):
        self.code = code
        self.status = status
        super().__init__(code)


def canonicalize_definitions(definitions: dict[str, Any]) -> dict[str, list[dict]]:
    if type(definitions) is not dict:
        raise ValueError("schema definitions must be an object")
    entities = definitions.get("entities", [])
    relations = definitions.get("relations", [])
    if type(entities) is not list or type(relations) is not list:
        raise ValueError("schema entities and relations must be arrays")
    for label, rows in (("entity", entities), ("relation", relations)):
        if any(type(row) is not dict for row in rows):
            raise ValueError(f"schema {label} definitions must be objects")
        keys = [row.get("key") for row in rows]
        if any(type(key) is not str or not key for key in keys):
            raise ValueError(f"schema {label} keys must be nonempty strings")
        if len(keys) != len(set(keys)):
            raise ValueError(f"schema {label} keys must be unique")
    result = {
        "entities": sorted(deepcopy(entities), key=lambda row: row["key"]),
        "relations": sorted(deepcopy(relations), key=lambda row: row["key"]),
    }
    json.dumps(result, allow_nan=False)
    return result


def definitions_checksum(definitions: dict[str, Any]) -> str:
    canonical = canonicalize_definitions(definitions)
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _permission_level(collection: Collection, user) -> str:
    if collection.user_can_manage(user):
        return "MANAGE"
    if collection.user_can_edit(user):
        return "EDIT"
    return "VIEW"


def permission_level(collection: Collection, user) -> str:
    return _permission_level(collection, user)


def _permissions(level: str) -> dict[str, bool | str]:
    editable = level in {"EDIT", "MANAGE"}
    manageable = level == "MANAGE"
    return {
        "level": level,
        "can_create_draft": editable,
        "can_edit_definitions": editable,
        "can_validate": editable,
        "can_publish": manageable,
        "can_discard_draft": manageable,
        "can_restore": manageable,
        "can_view_history": True,
    }


def workspace_envelope(collection: Collection, user) -> dict[str, Any]:
    level = _permission_level(collection, user)
    published = CollectionSchemaVersion.objects.filter(collection=collection).first()
    published_definitions = (
        _published_definitions(published.definitions)
        if published is not None
        else {"entities": [], "relations": []}
    )
    draft_payload = None
    if level != "VIEW":
        draft = getattr(collection, "schema_draft", None)
        if draft is not None:
            definitions = canonicalize_definitions(draft.definitions)
            draft_payload = {
                "draft_id": str(draft.pk),
                "revision": draft.revision,
                "base_published_checksum": (
                    draft.base_version.checksum
                    if draft.base_version is not None
                    else ""
                ),
                "last_editor": draft.last_editor.get_username(),
                "updated_at": draft.updated_at.isoformat(),
                **definitions,
            }
    return {
        "collection_id": str(collection.pk),
        "permissions": _permissions(level),
        "published": {
            "version": published.version if published is not None else 0,
            "checksum": published.checksum if published is not None else "",
            **published_definitions,
        },
        "draft": draft_payload,
        "constraints": deepcopy(CONSTRAINTS),
    }


def _capabilities(kind: str) -> dict[str, Any]:
    fields = (
        [
            "name",
            "description",
            "aliases",
            "default_retrieval_weight",
            "default_suppression_policy",
            "default_suppression_threshold",
        ]
        if kind == "entity"
        else [
            "name",
            "description",
            "direction",
            "allowed_head_types",
            "allowed_tail_types",
        ]
    )
    return {"editable_fields": fields, "removable": True, "renameable": True}


def _published_definitions(definitions: dict[str, Any]) -> dict[str, list[dict]]:
    canonical = canonicalize_definitions(definitions)
    for kind in ("entities", "relations"):
        for row in canonical[kind]:
            row["change_state"] = "unchanged"
    return canonical


def _candidate_definitions(definitions: dict[str, Any]) -> dict[str, list[dict]]:
    return _published_definitions(definitions)


def _next_version(collection: Collection) -> int:
    current = (
        CollectionSchemaVersion.objects.filter(collection=collection)
        .order_by("-version")
        .values_list("version", flat=True)
        .first()
    )
    return (current or 0) + 1


def create_draft(collection: Collection, user) -> CollectionSchemaDraft:
    with transaction.atomic():
        Collection.objects.select_for_update().get(pk=collection.pk)
        existing = (
            CollectionSchemaDraft.objects.select_for_update()
            .filter(collection=collection)
            .first()
        )
        if existing is not None:
            return existing
        base = CollectionSchemaVersion.objects.filter(collection=collection).first()
        definitions = (
            _published_definitions(base.definitions)
            if base is not None
            else {"entities": [], "relations": []}
        )
        return CollectionSchemaDraft.objects.create(
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
        CollectionSchemaDraft.objects.select_for_update()
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
) -> CollectionSchemaDraft:
    if kind not in {"entity", "relation"}:
        raise ValueError("unsupported schema definition kind")
    if type(key) is not str or not key:
        raise SchemaOperationError("invalid_definition_key")
    attempted = values or {}
    if values is not None and type(values) is not dict:
        raise SchemaOperationError("invalid_definition")
    with transaction.atomic():
        Collection.objects.select_for_update().get(pk=collection.pk)
        draft = _locked_draft(
            collection,
            revision,
            kind=kind,
            key=key,
            attempted=attempted,
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


def _ontology_document(collection_id: int, version: int, definitions: dict) -> dict:
    canonical = _candidate_definitions(definitions)
    return {
        "version": f"0.0.{version}+collection.{collection_id}",
        "entity_types": [deepcopy(row["values"]) for row in canonical["entities"]],
        "relations": [deepcopy(row["values"]) for row in canonical["relations"]],
    }


def _validation_result_id(draft_id, revision: int, checksum: str) -> str:
    return sha256(f"{draft_id}:{revision}:{checksum}".encode("ascii")).hexdigest()


def diff_definitions(base: dict, candidate: dict) -> dict[str, dict[str, int]]:
    result = {}
    for group in ("entities", "relations"):
        before = {row["key"]: row for row in canonicalize_definitions(base)[group]}
        after = {row["key"]: row for row in canonicalize_definitions(candidate)[group]}
        result[group] = {
            "added": len(after.keys() - before.keys()),
            "changed": sum(
                before[key].get("values") != after[key].get("values")
                for key in before.keys() & after.keys()
            ),
            "removed": len(before.keys() - after.keys()),
        }
    return result


def _diff_summary(
    base, candidate_version: int, checksum: str, definitions: dict
) -> dict:
    base_definitions = (
        base.definitions if base is not None else {"entities": [], "relations": []}
    )
    counts = diff_definitions(base_definitions, definitions)
    return {
        "base_version": base.version if base is not None else 0,
        "base_checksum": base.checksum if base is not None else "",
        "candidate_version": candidate_version,
        "candidate_checksum": checksum,
        **counts,
    }


def validate_draft(collection: Collection, draft_id, revision: int) -> dict[str, Any]:
    draft = CollectionSchemaDraft.objects.filter(
        collection=collection, pk=draft_id
    ).first()
    if draft is None:
        raise SchemaOperationError("draft_not_found", status=404)
    if revision != draft.revision:
        raise SchemaRevisionConflict(revision, draft)
    candidate = _candidate_definitions(draft.definitions)
    checksum = definitions_checksum(candidate)
    issues = []
    try:
        from apps.knowledge_graph.services.ontology import load_ontology_yaml

        load_ontology_yaml(
            yaml.safe_dump(
                _ontology_document(collection.pk, _next_version(collection), candidate),
                sort_keys=True,
            )
        )
    except ValueError as exc:
        issues.append(
            {
                "code": "ontology_invalid",
                "location": "schema",
                "message": str(exc),
                "severity": "error",
            }
        )
    return {
        "identity": {
            "draft_id": str(draft.pk),
            "revision": draft.revision,
            "candidate_checksum": checksum,
            "result_id": _validation_result_id(draft.pk, draft.revision, checksum),
        },
        "issues": issues,
        "diff_summary": _diff_summary(
            draft.base_version,
            _next_version(collection),
            checksum,
            candidate,
        ),
    }


def publish_draft(
    collection: Collection,
    user,
    operation: dict[str, Any],
    revision: int | None,
) -> CollectionSchemaVersion:
    with transaction.atomic():
        Collection.objects.select_for_update().get(pk=collection.pk)
        draft = _locked_draft(collection, revision)
        if str(draft.pk) != str(operation.get("draft_id")):
            raise SchemaOperationError("draft_identity_mismatch", status=409)
        validation = validate_draft(collection, draft.pk, draft.revision)
        identity = validation["identity"]
        if validation["issues"]:
            raise SchemaOperationError("validation_failed", status=422)
        if (
            operation.get("revision") != draft.revision
            or operation.get("candidate_checksum") != identity["candidate_checksum"]
            or operation.get("validation_result_id") != identity["result_id"]
        ):
            raise SchemaOperationError("validation_identity_mismatch", status=409)
        version_number = _next_version(collection)
        definitions = _candidate_definitions(draft.definitions)
        from apps.knowledge_graph.services.ontology import (
            activate_collection_ontology,
            load_ontology_yaml,
        )

        ontology = load_ontology_yaml(
            yaml.safe_dump(
                _ontology_document(collection.pk, version_number, definitions),
                sort_keys=True,
            )
        )
        ontology_record = activate_collection_ontology(collection.pk, ontology)
        version = CollectionSchemaVersion.objects.create(
            collection=collection,
            version=version_number,
            checksum=identity["candidate_checksum"],
            definitions=definitions,
            ontology_version=ontology_record,
            published_by=user,
            summary=f"Published schema version {version_number}",
        )
        draft.delete()

        def schedule_rebuild():
            from apps.knowledge_graph.models import GraphRebuildRequest
            from apps.knowledge_graph.services.builds import create_rebuild_request

            try:
                create_rebuild_request(
                    scope_type=GraphRebuildRequest.ScopeType.COLLECTION,
                    scope_id=collection.pk,
                )
            except Exception as exc:
                logger.warning(
                    "obs.kg.schema_rebuild_schedule_failed",
                    collection_id=collection.pk,
                    error_code=getattr(exc, "error_code", type(exc).__name__.lower()),
                )

        transaction.on_commit(schedule_rebuild)
        return version


def discard_draft(collection: Collection, draft_id, revision: int | None) -> None:
    with transaction.atomic():
        Collection.objects.select_for_update().get(pk=collection.pk)
        draft = _locked_draft(collection, revision)
        if str(draft.pk) != str(draft_id):
            raise SchemaOperationError("draft_identity_mismatch", status=409)
        draft.delete()


def draft_diff(collection: Collection) -> dict[str, Any]:
    draft = CollectionSchemaDraft.objects.filter(collection=collection).first()
    if draft is None:
        raise SchemaOperationError("draft_not_found", status=404)
    candidate = _candidate_definitions(draft.definitions)
    checksum = definitions_checksum(candidate)
    return _diff_summary(
        draft.base_version,
        _next_version(collection),
        checksum,
        candidate,
    )


def version_diff(collection: Collection, version: int) -> dict[str, Any]:
    candidate = CollectionSchemaVersion.objects.filter(
        collection=collection, version=version
    ).first()
    if candidate is None:
        raise SchemaOperationError("version_not_found", status=404)
    base = CollectionSchemaVersion.objects.filter(
        collection=collection, version__lt=version
    ).first()
    return _diff_summary(
        base, candidate.version, candidate.checksum, candidate.definitions
    )


def history_page(collection: Collection, cursor: str | None = None) -> dict[str, Any]:
    try:
        offset = max(int(cursor or 0), 0)
    except ValueError as exc:
        raise SchemaOperationError("invalid_cursor") from exc
    page_size = 50
    rows = list(
        CollectionSchemaVersion.objects.filter(collection=collection).order_by(
            "-version"
        )[offset : offset + page_size + 1]
    )
    has_more = len(rows) > page_size
    rows = rows[:page_size]
    return {
        "versions": [
            {
                "version": row.version,
                "checksum": row.checksum,
                "published_at": row.published_at.isoformat(),
                "summary": row.summary,
            }
            for row in rows
        ],
        "next_cursor": str(offset + page_size) if has_more else None,
        "has_more": has_more,
    }


def _restore_challenge(collection_id: int, version: int, draft) -> str:
    return sha256(
        f"restore:{collection_id}:{version}:{draft.pk}:{draft.revision}".encode("ascii")
    ).hexdigest()


def restore_version(collection: Collection, user, version: int):
    with transaction.atomic():
        Collection.objects.select_for_update().get(pk=collection.pk)
        source = CollectionSchemaVersion.objects.filter(
            collection=collection, version=version
        ).first()
        if source is None:
            raise SchemaOperationError("version_not_found", status=404)
        existing = (
            CollectionSchemaDraft.objects.select_for_update()
            .filter(collection=collection)
            .first()
        )
        if existing is not None:
            return {
                "challenge_token": _restore_challenge(collection.pk, version, existing),
                "existing_draft_revision": existing.revision,
                "existing_draft_id": str(existing.pk),
                "last_editor": existing.last_editor.get_username(),
            }
        CollectionSchemaDraft.objects.create(
            collection=collection,
            base_version=CollectionSchemaVersion.objects.filter(
                collection=collection
            ).first(),
            definitions=_published_definitions(source.definitions),
            last_editor=user,
        )
        return None


def replace_with_version(
    collection: Collection,
    user,
    version: int,
    challenge_token: str,
    revision: int | None,
) -> None:
    with transaction.atomic():
        Collection.objects.select_for_update().get(pk=collection.pk)
        draft = _locked_draft(collection, revision)
        if challenge_token != _restore_challenge(collection.pk, version, draft):
            raise SchemaOperationError("invalid_challenge")
        source = CollectionSchemaVersion.objects.filter(
            collection=collection, version=version
        ).first()
        if source is None:
            raise SchemaOperationError("version_not_found", status=404)
        draft.delete()
        CollectionSchemaDraft.objects.create(
            collection=collection,
            base_version=CollectionSchemaVersion.objects.filter(
                collection=collection
            ).first(),
            definitions=_published_definitions(source.definitions),
            last_editor=user,
        )


def write_generated_draft(run_id, definitions, statistics):
    from django.db import transaction

    from apps.collections.models import (
        CollectionSchemaDraft,
        CollectionSchemaGenerationRun,
    )

    canonical = canonicalize_definitions(definitions)
    with transaction.atomic():
        run = (
            CollectionSchemaGenerationRun.objects.select_for_update()
            .select_related("collection")
            .get(pk=run_id)
        )
        Collection.objects.select_for_update().get(pk=run.collection_id)
        if run.status != CollectionSchemaGenerationRun.Status.RUNNING:
            raise ValueError("schema generation run must be running")
        current = (
            CollectionSchemaDraft.objects.select_for_update()
            .filter(collection=run.collection)
            .first()
        )
        current_revision = current.revision if current is not None else None
        if current_revision != run.base_draft_revision:
            raise SchemaGenerationDraftConflict("draft_conflict")
        if current is None:
            if run.requested_by is None:
                raise ValueError("schema generation run requester is unavailable")
            draft = CollectionSchemaDraft.objects.create(
                collection=run.collection,
                base_version=run.collection.schema_versions.first(),
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
