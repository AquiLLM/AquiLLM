"""Schema validation and immutable version publication."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from typing import Any

import yaml
from django.db import transaction

from apps.collections.models import (
    Collection,
    CollectionSchemaVersion,
)

from . import schema as core
from .schema import (
    SchemaOperationError,
    SchemaRevisionConflict,
    _candidate_definitions,
    _locked_draft,
    canonicalize_definitions,
    definitions_checksum,
    logger,
)


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
    draft = core.CollectionSchemaDraft.objects.filter(
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
        from lib.knowledge_graph.query_extractor.ontology_payload import (
            ontology_definition_payload,
        )

        definition = load_ontology_yaml(
            yaml.safe_dump(
                _ontology_document(
                    collection.pk, core._next_version(collection), candidate
                ),
                sort_keys=True,
            )
        )
        ontology_definition_payload(definition)
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
            core._next_version(collection),
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
        locked_collection = core.Collection.objects.select_for_update().get(
            pk=collection.pk
        )
        draft = _locked_draft(collection, revision)
        if str(draft.pk) != str(operation.get("draft_id")):
            raise SchemaOperationError("draft_identity_mismatch", status=409)
        validation = core.validate_draft(collection, draft.pk, draft.revision)
        identity = validation["identity"]
        if validation["issues"]:
            raise SchemaOperationError("validation_failed", status=422)
        if (
            operation.get("revision") != draft.revision
            or operation.get("candidate_checksum") != identity["candidate_checksum"]
            or operation.get("validation_result_id") != identity["result_id"]
        ):
            raise SchemaOperationError("validation_identity_mismatch", status=409)
        definitions = _candidate_definitions(draft.definitions)
        from apps.knowledge_graph.services.ontology import (
            activate_collection_ontology,
            load_ontology_yaml,
        )

        version = (
            core.CollectionSchemaVersion.objects.select_for_update()
            .select_related("ontology_version")
            .filter(
                collection=collection,
                checksum=identity["candidate_checksum"],
            )
            .first()
        )
        if version is None:
            version_number = core._next_version(collection)
            ontology = load_ontology_yaml(
                yaml.safe_dump(
                    _ontology_document(collection.pk, version_number, definitions),
                    sort_keys=True,
                )
            )
            ontology_record = activate_collection_ontology(collection.pk, ontology)
            version = core.CollectionSchemaVersion.objects.create(
                collection=collection,
                version=version_number,
                checksum=identity["candidate_checksum"],
                definitions=definitions,
                ontology_version=ontology_record,
                published_by=user,
                summary=f"Published schema version {version_number}",
            )
        else:
            metadata = version.ontology_version.metadata
            yaml_snapshot = metadata.get("yaml") if type(metadata) is dict else None
            if type(yaml_snapshot) is not str:
                raise SchemaOperationError("published_ontology_snapshot_invalid")
            ontology_record = activate_collection_ontology(
                collection.pk,
                load_ontology_yaml(yaml_snapshot),
            )
            if ontology_record.pk != version.ontology_version_id:
                raise SchemaOperationError("published_ontology_identity_mismatch")

        head_changed = locked_collection.current_schema_version_id != version.pk
        locked_collection.current_schema_version = version
        locked_collection.save(update_fields=("current_schema_version",))
        collection.current_schema_version = version
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

        if head_changed:
            transaction.on_commit(schedule_rebuild)
        return version
