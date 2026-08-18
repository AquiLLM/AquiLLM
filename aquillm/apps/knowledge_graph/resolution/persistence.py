"""Transactional persistence boundary for pure document resolution output."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from hashlib import sha256
from math import isfinite
from uuid import UUID

from .coreference import (
    MAX_DOCUMENT_MENTIONS,
    ClusterMembership,
    PairDecision,
    ResolutionResult,
    ResolvedCluster,
    resolution_input_fingerprint,
    resolution_result_checksum,
)

_HASH = re.compile(r"[0-9a-f]{64}")
_COMMIT_FIELDS = frozenset(
    (
        "version",
        "resolver_version",
        "ontology_checksum",
        "source_mention_count",
        "source_mention_fingerprint",
        "document_entity_count",
        "membership_count",
        "result_checksum",
    )
)
_SOURCE_FIELDS = (
    "id",
    "artifact_id",
    "document_id",
    "chunk_id",
    "start",
    "end",
    "position_basis",
    "raw_text",
    "normalized_text",
    "entity_type",
    "extraction_confidence",
    "content_object_type_id",
    "content_object_id",
    "metadata",
)


class ResolutionPersistenceError(RuntimeError):
    """Raised when a resolution write cannot preserve its immutable snapshot."""


def _is_count(value: object) -> bool:
    return type(value) is int and value >= 0


def _is_hash(value: object) -> bool:
    return type(value) is str and _HASH.fullmatch(value) is not None


def resolution_commit_is_valid(
    marker: object,
    *,
    resolver_version: str,
    ontology_checksum: str,
    source_mention_count: int,
    source_mention_fingerprint: str,
    document_entity_count: int,
    membership_count: int,
    result_checksum: str,
) -> bool:
    """Validate the complete resolution commit marker against persisted state."""

    return bool(
        isinstance(marker, Mapping)
        and frozenset(marker) == _COMMIT_FIELDS
        and type(resolver_version) is str
        and _is_hash(ontology_checksum)
        and _is_count(source_mention_count)
        and _is_hash(source_mention_fingerprint)
        and _is_count(document_entity_count)
        and _is_count(membership_count)
        and _is_hash(result_checksum)
        and type(marker.get("version")) is int
        and marker.get("version") == 1
        and type(marker.get("resolver_version")) is str
        and marker.get("resolver_version") == resolver_version
        and _is_hash(marker.get("ontology_checksum"))
        and marker.get("ontology_checksum") == ontology_checksum
        and _is_count(marker.get("source_mention_count"))
        and marker.get("source_mention_count") == source_mention_count
        and _is_hash(marker.get("source_mention_fingerprint"))
        and marker.get("source_mention_fingerprint") == source_mention_fingerprint
        and _is_count(marker.get("document_entity_count"))
        and marker.get("document_entity_count") == document_entity_count
        and _is_count(marker.get("membership_count"))
        and marker.get("membership_count") == membership_count
        and marker.get("membership_count") == marker.get("source_mention_count")
        and _is_hash(marker.get("result_checksum"))
        and marker.get("result_checksum") == result_checksum
    )


def _record_value(record: object, field: str) -> object:
    if isinstance(record, Mapping):
        return record.get(field)
    return getattr(record, field, None)


def _json_value(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("mention fingerprint values must be finite")
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("mention metadata keys must be strings")
        return {key: _json_value(value[key]) for key in sorted(value)}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    raise ValueError(f"unsupported mention fingerprint value: {type(value).__name__}")


def source_mention_fingerprint(mentions) -> str:
    """Hash every persisted raw-mention field in stable primary-key order."""

    records = tuple(mentions)
    if len(records) > MAX_DOCUMENT_MENTIONS:
        raise ValueError(f"document mention cap exceeded ({MAX_DOCUMENT_MENTIONS})")
    normalized = [
        {field: _json_value(_record_value(record, field)) for field in _SOURCE_FIELDS}
        for record in records
    ]
    try:
        normalized.sort(key=lambda record: (str(type(record["id"])), record["id"]))
    except TypeError:
        normalized.sort(key=lambda record: repr(record["id"]))
    identifiers = [str(record["id"]) for record in normalized]
    if any(identifier in {"", "None"} for identifier in identifiers):
        raise ValueError("source mentions require stable primary keys")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("source mention primary keys must be unique")
    encoded = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _revalidate_immutable_result(result: object) -> ResolutionResult:
    try:
        if type(result) is not ResolutionResult:
            raise ValueError("result must be an exact ResolutionResult")
        if (
            type(result.mention_ids) is not tuple
            or len(result.mention_ids) > MAX_DOCUMENT_MENTIONS
        ):
            raise ValueError("result mention IDs must be a bounded exact tuple")
        if type(result.clusters) is not tuple:
            raise ValueError("result clusters must be an exact tuple")
        if type(result.decisions) is not tuple or len(result.decisions) > (
            len(result.mention_ids) * (len(result.mention_ids) - 1) // 2
        ):
            raise ValueError("result decisions must be a bounded exact tuple")
        for cluster in result.clusters:
            if type(cluster) is not ResolvedCluster:
                raise ValueError("clusters must contain exact ResolvedCluster values")
            if type(cluster.mention_ids) is not tuple:
                raise ValueError("cluster mention IDs must be an exact tuple")
            if type(cluster.memberships) is not tuple:
                raise ValueError("cluster memberships must be an exact tuple")
            for membership in cluster.memberships:
                if type(membership) is not ClusterMembership:
                    raise ValueError(
                        "memberships must contain exact ClusterMembership values"
                    )
                membership.__post_init__()
            cluster.__post_init__()
        for decision in result.decisions:
            if type(decision) is not PairDecision:
                raise ValueError("decisions must contain exact PairDecision values")
            decision.__post_init__()
        result.__post_init__()
    except (AttributeError, TypeError, ValueError) as exc:
        raise ResolutionPersistenceError(
            "resolution result failed immutable identity validation"
        ) from exc
    return result


def _validate_destination(artifact, run, result: ResolutionResult) -> None:
    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun

    if run.artifact_id != artifact.pk:
        raise ResolutionPersistenceError(
            "build run must be owned by the destination artifact"
        )
    if artifact.scope_type != GraphArtifact.ScopeType.DOCUMENT:
        raise ResolutionPersistenceError(
            "document resolution requires a document artifact"
        )
    if artifact.status != GraphArtifact.Status.BUILDING:
        raise ResolutionPersistenceError("destination artifact must be building")
    if run.status != GraphBuildRun.Status.RUNNING:
        raise ResolutionPersistenceError("destination build run must be running")
    if run.stage not in {
        GraphBuildRun.Stage.EXTRACTION,
        GraphBuildRun.Stage.RESOLUTION,
    }:
        raise ResolutionPersistenceError(
            "destination build run must be in extraction or resolution stage"
        )
    for field in (
        "scope_type",
        "scope_id",
        "source_hash",
        "ontology_version",
        "extractor_version",
        "resolver_version",
        "filter_policy_version",
    ):
        if getattr(run, field) != getattr(artifact, field):
            raise ResolutionPersistenceError(
                f"build run {field} does not match destination artifact"
            )
    result = _revalidate_immutable_result(result)
    if result.resolver_version != artifact.resolver_version:
        raise ResolutionPersistenceError(
            "result resolver version does not match destination artifact"
        )
    if resolution_result_checksum(result) != result.checksum:
        raise ResolutionPersistenceError("resolution result checksum is invalid")


def _validate_source_snapshot(artifact, result, mention_records) -> str:
    document_ids = {
        str(_record_value(record, "document_id")) for record in mention_records
    }
    if document_ids and document_ids != {str(artifact.scope_id)}:
        raise ResolutionPersistenceError(
            "source mentions do not belong to the destination document"
        )
    source_ids = tuple(str(_record_value(record, "id")) for record in mention_records)
    if len(source_ids) != len(set(source_ids)):
        raise ResolutionPersistenceError("source mention primary keys are duplicated")
    if type(result.mention_ids) is not tuple or not all(
        type(mention_id) is str for mention_id in result.mention_ids
    ):
        raise ResolutionPersistenceError(
            "resolution result mention IDs must be exact strings"
        )
    if set(source_ids) != set(result.mention_ids):
        raise ResolutionPersistenceError(
            "resolution result does not partition the persisted source mentions"
        )
    if not _is_hash(getattr(result, "input_fingerprint", None)):
        raise ResolutionPersistenceError(
            "resolution result input fingerprint must be an exact lowercase "
            "SHA-256 digest"
        )
    if resolution_input_fingerprint(mention_records) != result.input_fingerprint:
        raise ResolutionPersistenceError(
            "resolution result does not match current mention source context"
        )
    return source_mention_fingerprint(mention_records)


def _cluster_methods(cluster) -> list[str]:
    methods = {
        membership.method
        for membership in cluster.memberships
        if membership.method != "root"
    }
    return sorted(methods or {"root"})


def _entity_row_identity_is_exact(row) -> bool:
    metadata = getattr(row, "metadata", None)
    methods = metadata.get("methods") if type(metadata) is dict else None
    return bool(
        all(
            type(getattr(row, field_name, None)) is str
            for field_name in (
                "cluster_key",
                "label",
                "normalized_label",
                "version_signature",
                "entity_type",
                "identifier",
            )
        )
        and type(metadata) is dict
        and type(metadata.get("resolver_version")) is str
        and type(methods) is list
        and all(type(method) is str for method in methods)
        and _is_hash(metadata.get("result_checksum"))
    )


def _link_row_identity_is_exact(row) -> bool:
    metadata = getattr(row, "metadata", None)
    return bool(
        type(getattr(row.document_entity, "cluster_key", None)) is str
        and all(
            type(getattr(row, field_name, None)) is str
            for field_name in (
                "method",
                "resolver_version",
                "parent_mention_id",
                "reason",
            )
        )
        and type(metadata) is dict
        and _is_hash(metadata.get("result_checksum"))
    )


def _resolution_rows_match(result, entity_rows, link_rows) -> bool:
    try:
        result = _revalidate_immutable_result(result)
    except ResolutionPersistenceError:
        return False
    if len(entity_rows) != len(result.clusters) or len(link_rows) != len(
        result.mention_ids
    ):
        return False
    active_entity_rows = tuple(
        row for row in entity_rows if row.status == row.Status.ACTIVE
    )
    active_link_rows = tuple(
        row for row in link_rows if row.status == row.Status.ACTIVE
    )
    if not all(_entity_row_identity_is_exact(row) for row in active_entity_rows):
        return False
    if not all(_link_row_identity_is_exact(row) for row in active_link_rows):
        return False

    expected_entities = {
        cluster.cluster_key: (
            cluster.label,
            cluster.normalized_label,
            cluster.version_signature,
            cluster.entity_type,
            cluster.identifier,
            {
                "resolver_version": result.resolver_version,
                "methods": _cluster_methods(cluster),
                "resolution_confidence": cluster.confidence,
                "result_checksum": result.checksum,
            },
        )
        for cluster in result.clusters
    }
    actual_entities = {
        row.cluster_key: (
            row.label,
            row.normalized_label,
            row.version_signature,
            row.entity_type,
            row.identifier,
            row.metadata,
        )
        for row in active_entity_rows
    }
    expected_links = {
        (
            cluster.cluster_key,
            membership.mention_id,
            membership.method,
            result.resolver_version,
            membership.parent_mention_id or "",
            membership.reason,
            result.checksum,
        )
        for cluster in result.clusters
        for membership in cluster.memberships
    }
    actual_links = {
        (
            row.document_entity.cluster_key,
            str(row.mention_id),
            row.method,
            row.resolver_version,
            row.parent_mention_id,
            row.reason,
            (
                row.metadata.get("result_checksum")
                if isinstance(row.metadata, dict)
                else None
            ),
        )
        for row in active_link_rows
    }
    return actual_entities == expected_entities and actual_links == expected_links


def _existing_resolution(
    *,
    artifact,
    result,
    stats,
    ontology_checksum,
    source_count,
    source_fingerprint,
):
    from apps.knowledge_graph.models import DocumentEntity, DocumentEntityMention

    entity_rows = tuple(
        DocumentEntity.objects.filter(artifact=artifact).order_by("cluster_key")
    )
    link_rows = tuple(
        DocumentEntityMention.objects.select_related("document_entity")
        .filter(document_entity__artifact=artifact)
        .order_by("mention_id")
    )
    marker = stats.get("resolution_commit")
    if marker is None:
        if entity_rows or link_rows:
            raise ResolutionPersistenceError(
                "destination contains resolution rows without a commit marker"
            )
        return None
    if not resolution_commit_is_valid(
        marker,
        resolver_version=result.resolver_version,
        ontology_checksum=ontology_checksum,
        source_mention_count=source_count,
        source_mention_fingerprint=source_fingerprint,
        document_entity_count=len(result.clusters),
        membership_count=len(result.mention_ids),
        result_checksum=result.checksum,
    ):
        raise ResolutionPersistenceError("existing resolution commit marker is invalid")
    if not _resolution_rows_match(result, entity_rows, link_rows):
        raise ResolutionPersistenceError(
            "persisted resolution rows do not match their commit marker"
        )
    return entity_rows


def _write_resolution_rows(*, artifact, result, mentions_by_id):
    from apps.knowledge_graph.models import DocumentEntity, DocumentEntityMention

    entity_rows = [
        DocumentEntity(
            artifact=artifact,
            document_id=artifact.scope_id,
            cluster_key=cluster.cluster_key,
            label=cluster.label,
            normalized_label=cluster.normalized_label,
            version_signature=cluster.version_signature,
            entity_type=cluster.entity_type,
            identifier=cluster.identifier,
            status=DocumentEntity.Status.ACTIVE,
            metadata={
                "resolver_version": result.resolver_version,
                "methods": _cluster_methods(cluster),
                "resolution_confidence": cluster.confidence,
                "result_checksum": result.checksum,
            },
        )
        for cluster in result.clusters
    ]
    entities_by_key = {row.cluster_key: row for row in entity_rows}
    if len(entities_by_key) != len(entity_rows):
        raise ResolutionPersistenceError("resolution cluster keys are not unique")
    DocumentEntity.objects.bulk_create(entity_rows)
    link_rows = [
        DocumentEntityMention(
            document_entity=entities_by_key[cluster.cluster_key],
            mention=mentions_by_id[membership.mention_id],
            status=DocumentEntityMention.Status.ACTIVE,
            method=membership.method,
            resolver_version=result.resolver_version,
            parent_mention_id=membership.parent_mention_id or "",
            reason=membership.reason,
            metadata={"result_checksum": result.checksum},
        )
        for cluster in result.clusters
        for membership in cluster.memberships
    ]
    DocumentEntityMention.objects.bulk_create(link_rows)
    return tuple(sorted(entity_rows, key=lambda row: row.cluster_key)), len(link_rows)


def persist_document_resolution(artifact_id, build_run_id, result):
    """Persist a complete result without taking over build lifecycle transitions."""

    from django.db import transaction

    from apps.knowledge_graph.extraction.pipeline import extraction_commit_is_valid
    from apps.knowledge_graph.models import (
        EntityMention,
        GraphArtifact,
        GraphBuildRun,
        RelationMention,
    )

    with transaction.atomic():
        artifact = GraphArtifact.objects.select_for_update().get(pk=artifact_id)
        run = GraphBuildRun.objects.select_for_update().get(pk=build_run_id)
        _validate_destination(artifact, run, result)
        mentions = tuple(
            EntityMention.objects.select_for_update()
            .select_related("chunk")
            .filter(artifact=artifact)
            .order_by("pk")
        )
        source_count = len(mentions)
        relation_count = RelationMention.objects.filter(artifact=artifact).count()
        stats = dict(run.stats) if isinstance(run.stats, dict) else {}
        if not extraction_commit_is_valid(
            run,
            entity_count=source_count,
            relation_count=relation_count,
        ):
            raise ResolutionPersistenceError(
                "document resolution requires a valid extraction commit"
            )
        if result.ontology_checksum != stats["ontology_checksum"]:
            raise ResolutionPersistenceError(
                "result ontology checksum does not match extraction snapshot"
            )
        source_fingerprint = _validate_source_snapshot(artifact, result, mentions)
        ontology_checksum = stats["ontology_checksum"]
        existing = _existing_resolution(
            artifact=artifact,
            result=result,
            stats=stats,
            ontology_checksum=ontology_checksum,
            source_count=source_count,
            source_fingerprint=source_fingerprint,
        )
        if existing is not None:
            return existing
        mentions_by_id = {str(mention.pk): mention for mention in mentions}
        entity_rows, membership_count = _write_resolution_rows(
            artifact=artifact,
            result=result,
            mentions_by_id=mentions_by_id,
        )
        marker = {
            "version": 1,
            "resolver_version": result.resolver_version,
            "ontology_checksum": ontology_checksum,
            "source_mention_count": source_count,
            "source_mention_fingerprint": source_fingerprint,
            "document_entity_count": len(entity_rows),
            "membership_count": membership_count,
            "result_checksum": result.checksum,
        }
        if not resolution_commit_is_valid(
            marker,
            resolver_version=result.resolver_version,
            ontology_checksum=ontology_checksum,
            source_mention_count=source_count,
            source_mention_fingerprint=source_fingerprint,
            document_entity_count=len(entity_rows),
            membership_count=membership_count,
            result_checksum=result.checksum,
        ):
            raise ResolutionPersistenceError("generated resolution marker is invalid")
        run.stats = {**stats, "resolution_commit": marker}
        run.save(update_fields=["stats"])
        return entity_rows


__all__ = [
    "ResolutionPersistenceError",
    "persist_document_resolution",
    "resolution_commit_is_valid",
    "source_mention_fingerprint",
]
