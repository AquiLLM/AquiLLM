"""Deterministic, evidence-backed collection graph assembly.

The pure planner in this module deliberately has no Django or model-provider
imports.  The ORM boundary below it locks and revalidates the immutable Task 9
snapshot before persisting or activating a collection artifact.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from hashlib import sha256
from math import isfinite
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from apps.knowledge_graph.services.ontology import OntologyDefinition


ASSEMBLY_VERSION = "collection-assembly-v1"
ASSEMBLY_V1_MAX_ENTITIES = 50_000
ASSEMBLY_V1_MAX_LINKS = 250_000
ASSEMBLY_V1_MAX_RELATIONS = 50_000
ASSEMBLY_V1_MAX_EVIDENCE = 200_000
ASSEMBLY_V1_MAX_ORPHAN_ENTITIES = ASSEMBLY_V1_MAX_ENTITIES
_ASSEMBLY_INSERT_BATCH_SIZE = 1_000
_ENDPOINT_ID_BATCH_SIZE = 5_000
_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SAFE_TOKEN_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,127}$")
_ACTIVE = "active"


class CollectionGraphAssemblyError(RuntimeError):
    """Raised when assembly cannot preserve its locked input snapshot."""


class EvidenceDisposition(StrEnum):
    PROMOTED = "promoted"
    SUPPRESSED = "suppressed"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class AssemblyConfig:
    """Bounded, checksum-addressed collection assembly policy."""

    version: str = ASSEMBLY_VERSION
    max_entities: int = ASSEMBLY_V1_MAX_ENTITIES
    max_links: int = ASSEMBLY_V1_MAX_LINKS
    # V1 materializes its deterministic projection before batched persistence.
    # Keep a hard operational envelope until a streaming planner lands.
    max_relations: int = ASSEMBLY_V1_MAX_RELATIONS
    max_evidence: int = ASSEMBLY_V1_MAX_EVIDENCE
    max_orphan_entities: int = ASSEMBLY_V1_MAX_ORPHAN_ENTITIES
    max_orphan_ratio: float = 1.0
    generic_identity_relations: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                "equivalent_to",
                "acronym_for",
                "alias_of",
                "identical_to",
                "identity",
                "is",
                "same_as",
                "same_entity_as",
            }
        )
    )
    generic_relation_types: frozenset[str] = field(
        default_factory=lambda: frozenset({"related_to"})
    )

    def __post_init__(self) -> None:
        if (
            type(self.version) is not str
            or not self.version
            or self.version != self.version.strip()
            or len(self.version) > 128
            or "\x00" in self.version
        ):
            raise ValueError("assembly version must be a safe nonempty string")
        upper_bounds = {
            "max_entities": ASSEMBLY_V1_MAX_ENTITIES,
            "max_links": ASSEMBLY_V1_MAX_LINKS,
            "max_relations": ASSEMBLY_V1_MAX_RELATIONS,
            "max_evidence": ASSEMBLY_V1_MAX_EVIDENCE,
            "max_orphan_entities": ASSEMBLY_V1_MAX_ORPHAN_ENTITIES,
        }
        for field_name, upper_bound in upper_bounds.items():
            value = getattr(self, field_name)
            minimum = 0 if field_name == "max_orphan_entities" else 1
            if type(value) is not int or not minimum <= value <= upper_bound:
                raise ValueError(f"{field_name} is outside its bounded range")
        if (
            isinstance(self.max_orphan_ratio, bool)
            or not isinstance(self.max_orphan_ratio, (int, float))
            or not isfinite(self.max_orphan_ratio)
            or not 0 <= self.max_orphan_ratio <= 1
        ):
            raise ValueError("max_orphan_ratio must be finite and in [0, 1]")
        for field_name in (
            "generic_identity_relations",
            "generic_relation_types",
        ):
            values = getattr(self, field_name)
            if type(values) is not frozenset or any(
                type(value) is not str or not _SAFE_TOKEN_PATTERN.fullmatch(value)
                for value in values
            ):
                raise ValueError(f"{field_name} must contain canonical tokens")
        if self.generic_identity_relations.intersection(self.generic_relation_types):
            raise ValueError("generic relation policies must not overlap")


def assembly_config_checksum(config: AssemblyConfig) -> str:
    """Return the typed SHA-256 identity of every assembly policy input."""

    if type(config) is not AssemblyConfig:
        raise ValueError("assembly config must be an exact AssemblyConfig")
    config.__post_init__()
    payload = {
        "version": config.version,
        "max_entities": config.max_entities,
        "max_links": config.max_links,
        "max_relations": config.max_relations,
        "max_evidence": config.max_evidence,
        "max_orphan_entities": config.max_orphan_entities,
        "max_orphan_ratio": float(config.max_orphan_ratio),
        "generic_identity_relations": sorted(config.generic_identity_relations),
        "generic_relation_types": sorted(config.generic_relation_types),
    }
    return _content_checksum(payload)


@dataclass(frozen=True, slots=True)
class AssemblyEvidenceInput:
    """One real RelationMention lifted through exact Task 8/9 mappings."""

    relation_mention_id: int
    document_artifact_id: int
    chunk_id: int | None
    head_mention_id: int
    tail_mention_id: int
    head_mapping_id: int | None
    tail_mapping_id: int | None
    head_collection_entity_id: int | None
    tail_collection_entity_id: int | None
    head_entity_type: str | None
    tail_entity_type: str | None
    head_status: str | None
    tail_status: str | None
    relation_type: str
    extraction_confidence: float
    head_cluster_key: str | None = None
    tail_cluster_key: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "relation_mention_id",
            "document_artifact_id",
            "chunk_id",
            "head_mention_id",
            "tail_mention_id",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value < 1:
                raise ValueError(
                    "relation evidence requires real chunk provenance and row IDs"
                )
        for field_name in (
            "head_mapping_id",
            "tail_mapping_id",
            "head_collection_entity_id",
            "tail_collection_entity_id",
        ):
            value = getattr(self, field_name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{field_name} must be a positive row ID or null")
        if (
            type(self.relation_type) is not str
            or not _SAFE_TOKEN_PATTERN.fullmatch(self.relation_type)
        ):
            raise ValueError("relation type must be a canonical token")
        for field_name in ("head_entity_type", "tail_entity_type"):
            value = getattr(self, field_name)
            if value is not None and (
                type(value) is not str or not _SAFE_TOKEN_PATTERN.fullmatch(value)
            ):
                raise ValueError(f"{field_name} must be a canonical token or null")
        confidence = self.extraction_confidence
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not isfinite(confidence)
            or not 0 <= confidence <= 1
        ):
            raise ValueError("extraction confidence must be finite and in [0, 1]")


RelationKey = tuple[int, str, int]


@dataclass(frozen=True, slots=True)
class PlannedRelation:
    source_entity_id: int
    relation_type: str
    target_entity_id: int
    support_count: int
    confidence: float
    evidence_mention_ids: tuple[int, ...]

    @property
    def key(self) -> RelationKey:
        return self.source_entity_id, self.relation_type, self.target_entity_id


@dataclass(frozen=True, slots=True)
class PlannedEvidence:
    relation_mention_id: int
    disposition: EvidenceDisposition
    reason: str
    relation_key: RelationKey | None
    head_mapping_id: int | None
    tail_mapping_id: int | None
    orientation: str


@dataclass(frozen=True, slots=True)
class CollectionAssemblyPlan:
    relations: tuple[PlannedRelation, ...]
    evidence: tuple[PlannedEvidence, ...]
    checksum: str


@dataclass(frozen=True, slots=True)
class AssemblyProjectionStats:
    entity_count: int
    relation_count: int
    evidence_count: int
    promoted_evidence_count: int
    suppressed_evidence_count: int
    rejected_evidence_count: int
    orphan_count: int
    orphan_ratio: float


def _content_checksum(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _ontology_relation_key(
    item: AssemblyEvidenceInput,
    ontology: OntologyDefinition,
    config: AssemblyConfig,
) -> tuple[RelationKey | None, EvidenceDisposition, str, str]:
    if item.relation_type in config.generic_identity_relations:
        return (
            None,
            EvidenceDisposition.SUPPRESSED,
            "generic_identity_relation",
            "head_to_tail",
        )
    if item.relation_type in config.generic_relation_types:
        return (
            None,
            EvidenceDisposition.SUPPRESSED,
            "generic_relation",
            "head_to_tail",
        )
    if (
        item.head_mapping_id is None
        or item.tail_mapping_id is None
        or item.head_collection_entity_id is None
        or item.tail_collection_entity_id is None
    ):
        return (
            None,
            EvidenceDisposition.REJECTED,
            "missing_active_mapping",
            "head_to_tail",
        )
    if item.head_status != _ACTIVE or item.tail_status != _ACTIVE:
        return (
            None,
            EvidenceDisposition.REJECTED,
            "inactive_endpoint",
            "head_to_tail",
        )
    if item.head_collection_entity_id == item.tail_collection_entity_id:
        return (
            None,
            EvidenceDisposition.SUPPRESSED,
            "self_loop",
            "head_to_tail",
        )
    relation = ontology.relations.get(item.relation_type)
    if relation is None:
        return (
            None,
            EvidenceDisposition.REJECTED,
            "relation_not_in_ontology",
            "head_to_tail",
        )
    forward_allowed = (
        item.head_entity_type in relation.allowed_head_types
        and item.tail_entity_type in relation.allowed_tail_types
    )
    reverse_allowed = (
        relation.direction == "undirected"
        and item.tail_entity_type in relation.allowed_head_types
        and item.head_entity_type in relation.allowed_tail_types
    )
    if not forward_allowed and not reverse_allowed:
        return (
            None,
            EvidenceDisposition.REJECTED,
            "ontology_endpoint_types_invalid",
            "head_to_tail",
        )
    source = item.head_collection_entity_id
    target = item.tail_collection_entity_id
    orientation = "head_to_tail"
    if relation.direction == "undirected":
        if (
            type(item.head_cluster_key) is not str
            or not _HASH_PATTERN.fullmatch(item.head_cluster_key)
            or type(item.tail_cluster_key) is not str
            or not _HASH_PATTERN.fullmatch(item.tail_cluster_key)
        ):
            raise ValueError(
                "undirected relation evidence requires stable collection cluster keys"
            )
        if item.tail_cluster_key < item.head_cluster_key:
            source, target = target, source
            orientation = "tail_to_head"
    return (
        (source, item.relation_type, target),
        EvidenceDisposition.PROMOTED,
        "ontology_valid_supported_evidence",
        orientation,
    )


def plan_collection_relations(
    evidence: tuple[AssemblyEvidenceInput, ...],
    ontology: OntologyDefinition,
    config: AssemblyConfig | None = None,
) -> CollectionAssemblyPlan:
    """Aggregate valid assertions while retaining every rejected extraction."""

    from apps.knowledge_graph.services.ontology import validate_ontology_definition

    if type(evidence) is not tuple:
        raise ValueError("assembly evidence must be an immutable tuple")
    config = AssemblyConfig() if config is None else config
    if type(config) is not AssemblyConfig:
        raise ValueError("assembly config must be an exact AssemblyConfig")
    config.__post_init__()
    try:
        validate_ontology_definition(ontology)
    except (TypeError, ValueError) as exc:
        raise ValueError("assembly requires an exact validated ontology") from exc
    if len(evidence) > config.max_evidence:
        raise ValueError("assembly evidence cap exceeded")
    mention_ids = tuple(item.relation_mention_id for item in evidence)
    if len(mention_ids) != len(set(mention_ids)):
        raise ValueError("duplicate relation mention evidence")
    if any(type(item) is not AssemblyEvidenceInput for item in evidence):
        raise ValueError("assembly evidence must use exact input records")

    planned_evidence: list[PlannedEvidence] = []
    support: dict[RelationKey, list[AssemblyEvidenceInput]] = {}
    for item in sorted(evidence, key=lambda value: value.relation_mention_id):
        item.__post_init__()
        key, disposition, reason, orientation = _ontology_relation_key(
            item, ontology, config
        )
        planned_evidence.append(
            PlannedEvidence(
                relation_mention_id=item.relation_mention_id,
                disposition=disposition,
                reason=reason,
                relation_key=key,
                head_mapping_id=item.head_mapping_id,
                tail_mapping_id=item.tail_mapping_id,
                orientation=orientation,
            )
        )
        if key is not None:
            support.setdefault(key, []).append(item)
    if len(support) > config.max_relations:
        raise ValueError("assembly relation cap exceeded")

    relations = tuple(
        PlannedRelation(
            source_entity_id=key[0],
            relation_type=key[1],
            target_entity_id=key[2],
            support_count=len(items),
            confidence=max(float(item.extraction_confidence) for item in items),
            evidence_mention_ids=tuple(
                sorted(item.relation_mention_id for item in items)
            ),
        )
        for key, items in sorted(support.items())
    )
    content = {
        "assembly_version": config.version,
        "config_checksum": assembly_config_checksum(config),
        "ontology_checksum": ontology.checksum,
        "relations": [
            {
                "key": list(item.key),
                "support_count": item.support_count,
                "confidence": item.confidence,
                "evidence_mention_ids": list(item.evidence_mention_ids),
            }
            for item in relations
        ],
        "evidence": [
            {
                "relation_mention_id": item.relation_mention_id,
                "disposition": item.disposition.value,
                "reason": item.reason,
                "relation_key": (
                    None if item.relation_key is None else list(item.relation_key)
                ),
                "head_mapping_id": item.head_mapping_id,
                "tail_mapping_id": item.tail_mapping_id,
                "orientation": item.orientation,
            }
            for item in planned_evidence
        ],
    }
    return CollectionAssemblyPlan(
        relations=relations,
        evidence=tuple(planned_evidence),
        checksum=_content_checksum(content),
    )


def validate_assembly_projection(
    plan: CollectionAssemblyPlan,
    *,
    active_entity_ids: frozenset[int],
    provenanced_entity_ids: frozenset[int],
    config: AssemblyConfig,
) -> AssemblyProjectionStats:
    """Validate bounded active topology independently of the persistence layer."""

    if type(plan) is not CollectionAssemblyPlan:
        raise ValueError("projection validation requires an exact assembly plan")
    if type(config) is not AssemblyConfig:
        raise ValueError("projection validation requires an exact assembly config")
    config.__post_init__()
    for label, values in (
        ("active entity", active_entity_ids),
        ("provenanced entity", provenanced_entity_ids),
    ):
        if type(values) is not frozenset or any(
            type(value) is not int or value < 1 for value in values
        ):
            raise ValueError(f"{label} IDs must be positive immutable row IDs")
    if len(active_entity_ids) > config.max_entities:
        raise ValueError("assembly entity cap exceeded")
    if len(plan.relations) > config.max_relations:
        raise ValueError("assembly relation cap exceeded")
    if len(plan.evidence) > config.max_evidence:
        raise ValueError("assembly evidence cap exceeded")
    missing_provenance = active_entity_ids.difference(provenanced_entity_ids)
    if missing_provenance:
        raise ValueError("active collection entities lack real chunk provenance")
    relation_endpoints = frozenset(
        endpoint
        for relation in plan.relations
        for endpoint in (relation.source_entity_id, relation.target_entity_id)
    )
    if not relation_endpoints.issubset(active_entity_ids):
        raise ValueError("active relation endpoint is not an active collection entity")
    orphans = active_entity_ids.difference(relation_endpoints)
    orphan_ratio = (
        0.0 if not active_entity_ids else len(orphans) / len(active_entity_ids)
    )
    if (
        len(orphans) > config.max_orphan_entities
        or orphan_ratio > config.max_orphan_ratio
    ):
        raise ValueError("collection graph orphan limits exceeded")
    disposition_counts = {
        disposition: sum(
            item.disposition is disposition for item in plan.evidence
        )
        for disposition in EvidenceDisposition
    }
    return AssemblyProjectionStats(
        entity_count=len(active_entity_ids),
        relation_count=len(plan.relations),
        evidence_count=len(plan.evidence),
        promoted_evidence_count=disposition_counts[EvidenceDisposition.PROMOTED],
        suppressed_evidence_count=disposition_counts[EvidenceDisposition.SUPPRESSED],
        rejected_evidence_count=disposition_counts[EvidenceDisposition.REJECTED],
        orphan_count=len(orphans),
        orphan_ratio=orphan_ratio,
    )


@dataclass(frozen=True, slots=True)
class AssemblyResult:
    artifact_id: int
    build_run_id: int
    source_hash: str
    plan_checksum: str
    marker_checksum: str
    stats: AssemblyProjectionStats


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or value < 1:
        raise CollectionGraphAssemblyError(f"{label} must be a positive integer")
    return value


def _source_hash(value: object) -> str:
    if type(value) is not str or not _HASH_PATTERN.fullmatch(value):
        raise CollectionGraphAssemblyError(
            "aggregate source signature must be a lowercase SHA-256 checksum"
        )
    return value


def _id_batches(values: Iterable[int]) -> Iterator[tuple[int, ...]]:
    """Yield stable endpoint-ID batches small enough for PostgreSQL predicates."""

    ordered = tuple(sorted(set(values)))
    if any(type(value) is not int or value < 1 for value in ordered):
        raise CollectionGraphAssemblyError("endpoint IDs must be positive integers")
    for offset in range(0, len(ordered), _ENDPOINT_ID_BATCH_SIZE):
        yield ordered[offset : offset + _ENDPOINT_ID_BATCH_SIZE]


def _config_payload(config: AssemblyConfig) -> dict[str, object]:
    return {
        "version": config.version,
        "max_entities": config.max_entities,
        "max_links": config.max_links,
        "max_relations": config.max_relations,
        "max_evidence": config.max_evidence,
        "max_orphan_entities": config.max_orphan_entities,
        "max_orphan_ratio": float(config.max_orphan_ratio),
        "generic_identity_relations": sorted(config.generic_identity_relations),
        "generic_relation_types": sorted(config.generic_relation_types),
    }


def _config_from_marker(marker: object) -> AssemblyConfig:
    if type(marker) is not dict or type(marker.get("config")) is not dict:
        raise CollectionGraphAssemblyError("assembly marker has no typed config")
    payload = marker["config"]
    try:
        if set(payload) != {
            "version",
            "max_entities",
            "max_links",
            "max_relations",
            "max_evidence",
            "max_orphan_entities",
            "max_orphan_ratio",
            "generic_identity_relations",
            "generic_relation_types",
        }:
            raise ValueError("unexpected assembly config fields")
        identity_values = payload["generic_identity_relations"]
        generic_values = payload["generic_relation_types"]
        if type(identity_values) is not list or type(generic_values) is not list:
            raise ValueError("assembly relation policies must be lists")
        return AssemblyConfig(
            version=payload["version"],
            max_entities=payload["max_entities"],
            max_links=payload["max_links"],
            max_relations=payload["max_relations"],
            max_evidence=payload["max_evidence"],
            max_orphan_entities=payload["max_orphan_entities"],
            max_orphan_ratio=payload["max_orphan_ratio"],
            generic_identity_relations=frozenset(identity_values),
            generic_relation_types=frozenset(generic_values),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CollectionGraphAssemblyError(
            "assembly marker config is invalid"
        ) from exc


def _resolve_config(run: object, config: AssemblyConfig | None) -> AssemblyConfig:
    if config is not None:
        if type(config) is not AssemblyConfig:
            raise CollectionGraphAssemblyError(
                "assembly config must be an exact AssemblyConfig"
            )
        config.__post_init__()
        resolved = config
    else:
        stats = run.stats if type(run.stats) is dict else {}
        marker = stats.get("collection_assembly_commit")
        resolved = AssemblyConfig() if marker is None else _config_from_marker(marker)
    if (
        resolved.version != run.assembly_version
        or assembly_config_checksum(resolved) != run.assembly_config_checksum
    ):
        raise CollectionGraphAssemblyError(
            "assembly config does not match the immutable build identity"
        )
    return resolved


def _resolve_ontology(artifact: object, ontology: object | None):
    from apps.knowledge_graph.models import OntologyVersion
    from apps.knowledge_graph.services.ontology import (
        load_ontology_yaml,
        validate_ontology_definition,
    )

    if ontology is None:
        record = (
            OntologyVersion.objects.select_for_update()
            .filter(
                kind=OntologyVersion.Kind.GRAPH,
                version=artifact.ontology_version,
                checksum=artifact.ontology_checksum,
            )
            .order_by("pk")
            .first()
        )
        metadata = None if record is None else record.metadata
        raw_yaml = metadata.get("yaml") if type(metadata) is dict else None
        if type(raw_yaml) is not str:
            raise CollectionGraphAssemblyError(
                "artifact ontology identity has no persisted definition"
            )
        ontology = load_ontology_yaml(raw_yaml)
    try:
        return validate_ontology_definition(
            ontology,
            expected_version=artifact.ontology_version,
            expected_checksum=artifact.ontology_checksum,
        )
    except (TypeError, ValueError) as exc:
        raise CollectionGraphAssemblyError(
            "ontology does not match the collection artifact identity"
        ) from exc


def _marker_hash(marker: dict[str, object]) -> str:
    content = {key: value for key, value in marker.items() if key != "marker_checksum"}
    return _content_checksum(content)


def _require_marker_hash(marker: object, label: str) -> dict[str, object]:
    if type(marker) is not dict:
        raise CollectionGraphAssemblyError(f"{label} marker is missing")
    checksum = marker.get("marker_checksum")
    if type(checksum) is not str or checksum != _marker_hash(marker):
        raise CollectionGraphAssemblyError(f"{label} marker checksum is invalid")
    return marker


def _relation_row_content(row: object) -> dict[str, object]:
    metadata = row.metadata if type(row.metadata) is dict else {}
    return {
        "artifact_id": row.artifact_id,
        "source_id": row.source_id,
        "relation_type": row.relation_type,
        "target_id": row.target_id,
        "status": row.status,
        "support_count": row.support_count,
        "confidence": row.confidence,
        "metadata": {
            key: value
            for key, value in metadata.items()
            if key != "row_audit_checksum"
        },
    }


def _relation_row_audit(row: object) -> str:
    return _content_checksum(_relation_row_content(row))


def _evidence_row_content(row: object) -> dict[str, object]:
    metadata = row.metadata if type(row.metadata) is dict else {}
    return {
        "artifact_id": row.artifact_id,
        "relation_id": row.relation_id,
        "relation_mention_id": row.relation_mention_id,
        "head_mapping_id": row.head_mapping_id,
        "tail_mapping_id": row.tail_mapping_id,
        "status": row.status,
        "reason": row.reason,
        "orientation": row.orientation,
        "ontology_checksum": row.ontology_checksum,
        "assembly_config_checksum": row.assembly_config_checksum,
        "metadata": {
            key: value
            for key, value in metadata.items()
            if key != "row_audit_checksum"
        },
    }


def _evidence_row_audit(row: object) -> str:
    return _content_checksum(_evidence_row_content(row))


def _validate_task9_lineage(
    artifact: object,
    run: object,
    manifest: tuple[object, ...],
    entities: tuple[object, ...],
    links: tuple[object, ...],
    *,
    visited_artifact_ids: frozenset[int] = frozenset(),
) -> str:
    from apps.knowledge_graph.resolution.collection import (
        CollectionResolutionPersistenceError,
        _collection_entity_row_audit,
        _collection_link_row_audit,
        _load_resolution_source_rows,
        _raw_relation_snapshot,
        _source_entity_fingerprint,
        _source_relation_fingerprint,
    )

    if artifact.pk in visited_artifact_ids:
        raise CollectionGraphAssemblyError("Task 9 lineage contains an artifact cycle")
    visited_artifact_ids = visited_artifact_ids | {artifact.pk}

    stats = run.stats if type(run.stats) is dict else {}
    resolution = stats.get("collection_resolution_commit")
    filter_commit = stats.get("filter_commit")
    if (resolution is None) == (filter_commit is None):
        raise CollectionGraphAssemblyError(
            "candidate requires exactly one Task 9 resolution/filter commit marker"
        )
    marker = resolution if resolution is not None else filter_commit
    if type(marker) is not dict:
        raise CollectionGraphAssemblyError("Task 9 lineage marker is invalid")
    assembly_committed = stats.get("collection_assembly_commit") is not None
    if assembly_committed:
        allowed_run_state = (
            (run.Stage.PERSISTENCE, run.Status.RUNNING),
            (run.Stage.COMPLETE, run.Status.SUCCEEDED),
        )
    elif resolution is not None:
        allowed_run_state = (
            (run.Stage.RESOLUTION, run.Status.RUNNING),
            (run.Stage.PERSISTENCE, run.Status.RUNNING),
        )
    else:
        allowed_run_state = (
            (run.Stage.FILTERING, run.Status.SUCCEEDED),
            (run.Stage.PERSISTENCE, run.Status.RUNNING),
        )
    if (run.stage, run.status) not in allowed_run_state:
        raise CollectionGraphAssemblyError(
            "build run stage/status is invalid for collection assembly"
        )
    if type(marker.get("version")) is not int or marker["version"] != 1:
        raise CollectionGraphAssemblyError("Task 9 lineage marker version is invalid")
    normal_marker_keys = {
        "version",
        "source_hash",
        "source_entity_fingerprint",
        "source_relation_fingerprint",
        "result_checksum",
        "filter_result_checksum",
        "filter_policy_checksum",
        "ontology_checksum",
        "resolution_config_checksum",
        "embedding_model_signature",
        "assembly_version",
        "assembly_config_checksum",
        "raw_relation_count",
        "raw_relation_fingerprint",
        "collection_entity_count",
        "automatic_assignment_count",
        "link_count",
    }
    filter_marker_keys = {
        "version",
        "policy_checksum",
        "ontology_checksum",
        "resolution_config_checksum",
        "filter_result_checksum",
        "source_artifact_id",
        "source_build_run_id",
        "source_task9_marker_checksum",
        "source_assembly_marker_checksum",
        "source_hash",
        "assembly_version",
        "assembly_config_checksum",
        "manifest_count",
        "entity_count",
        "link_count",
    }
    expected_marker_keys = (
        normal_marker_keys if resolution is not None else filter_marker_keys
    )
    if set(marker) != expected_marker_keys:
        raise CollectionGraphAssemblyError(
            "Task 9 lineage marker schema is not exact"
        )
    checksum_keys = {
        "source_hash",
        "ontology_checksum",
        "resolution_config_checksum",
        "filter_result_checksum",
        "assembly_config_checksum",
    }
    checksum_keys.add(
        "filter_policy_checksum" if resolution is not None else "policy_checksum"
    )
    if any(
        type(marker.get(key)) is not str
        or not _HASH_PATTERN.fullmatch(marker[key])
        for key in checksum_keys
    ):
        raise CollectionGraphAssemblyError(
            "Task 9 lineage marker has an invalid typed checksum"
        )
    if resolution is not None and any(
        type(marker.get(key)) is not str
        or not _HASH_PATTERN.fullmatch(marker[key])
        for key in (
            "source_entity_fingerprint",
            "source_relation_fingerprint",
            "raw_relation_fingerprint",
            "result_checksum",
        )
    ):
        raise CollectionGraphAssemblyError(
            "Task 9 resolution marker has an invalid result fingerprint"
        )
    if filter_commit is not None and (
        type(marker.get("source_build_run_id")) is not int
        or marker["source_build_run_id"] < 1
        or type(marker.get("source_task9_marker_checksum")) is not str
        or not _HASH_PATTERN.fullmatch(marker["source_task9_marker_checksum"])
        or (
            marker.get("source_assembly_marker_checksum") is not None
            and (
                type(marker["source_assembly_marker_checksum"]) is not str
                or not _HASH_PATTERN.fullmatch(
                    marker["source_assembly_marker_checksum"]
                )
            )
        )
    ):
        raise CollectionGraphAssemblyError(
            "filter rerun source commit identity is invalid"
        )
    count_keys = (
        (
            "collection_entity_count",
            "automatic_assignment_count",
            "link_count",
            "raw_relation_count",
        )
        if resolution is not None
        else ("entity_count", "link_count", "manifest_count")
    )
    if any(type(marker.get(key)) is not int or marker[key] < 0 for key in count_keys):
        raise CollectionGraphAssemblyError(
            "Task 9 lineage marker has an invalid typed count"
        )
    expected = {
        "source_hash": artifact.source_hash,
        "ontology_checksum": artifact.ontology_checksum,
        "resolution_config_checksum": artifact.resolution_config_checksum,
        "assembly_version": artifact.assembly_version,
        "assembly_config_checksum": artifact.assembly_config_checksum,
    }
    if resolution is not None:
        expected.update(
            {
                "filter_policy_checksum": artifact.filter_policy_checksum,
                "collection_entity_count": len(entities),
                "link_count": len(links),
            }
        )
    else:
        expected.update(
            {
                "policy_checksum": artifact.filter_policy_checksum,
                "manifest_count": len(manifest),
                "entity_count": len(entities),
                "link_count": len(links),
            }
        )
    if any(marker.get(key) != value for key, value in expected.items()):
        raise CollectionGraphAssemblyError(
            "Task 9 lineage marker does not match locked rows"
        )
    if resolution is not None:
        try:
            source_entities, source_relations = _load_resolution_source_rows(
                artifact,
                manifest,
                for_update=True,
            )
            raw_relation_count, raw_relation_fingerprint = _raw_relation_snapshot(
                manifest, for_update=True
            )
        except CollectionResolutionPersistenceError as exc:
            raise CollectionGraphAssemblyError(
                "Task 9 source snapshot failed exact validation"
            ) from exc
        if (
            marker["source_entity_fingerprint"]
            != _source_entity_fingerprint(source_entities)
            or marker["source_relation_fingerprint"]
            != _source_relation_fingerprint(source_relations)
            or marker["raw_relation_count"] != raw_relation_count
            or marker["raw_relation_fingerprint"] != raw_relation_fingerprint
            or marker["embedding_model_signature"]
            != artifact.embedding_model_signature
        ):
            raise CollectionGraphAssemblyError(
                "Task 9 source entity/relation fingerprint drift detected"
            )
    for row in entities:
        metadata = row.metadata if type(row.metadata) is dict else {}
        if (
            row.artifact_id != artifact.pk
            or str(row.collection_id) != artifact.scope_id
            or metadata.get("filter_policy_checksum")
            != artifact.filter_policy_checksum
            or metadata.get("filter_result_checksum")
            != marker["filter_result_checksum"]
            or metadata.get("row_audit_checksum")
            != _collection_entity_row_audit(row)
        ):
            raise CollectionGraphAssemblyError(
                "Task 9 collection entity audit is corrupt"
            )
    entity_ids = {row.pk for row in entities}
    automatic_source_ids: set[int] = set()
    for row in links:
        metadata = row.metadata if type(row.metadata) is dict else {}
        if (
            row.artifact_id != artifact.pk
            or row.collection_entity_id not in entity_ids
            or row.manifest_input.artifact_id != artifact.pk
            or metadata.get("filter_result_checksum")
            != marker["filter_result_checksum"]
            or metadata.get("row_audit_checksum") != _collection_link_row_audit(row)
        ):
            raise CollectionGraphAssemblyError(
                "Task 9 collection link audit is corrupt"
            )
        if row.outcome == row.Outcome.AUTOMATIC:
            if row.document_entity_id in automatic_source_ids:
                raise CollectionGraphAssemblyError(
                    "Task 9 automatic assignments do not partition source entities"
                )
            automatic_source_ids.add(row.document_entity_id)
    if (
        resolution is not None
        and marker["automatic_assignment_count"] != len(automatic_source_ids)
    ):
        raise CollectionGraphAssemblyError(
            "Task 9 automatic assignment count differs from locked links"
        )
    upstream_checksum = None
    if filter_commit is not None:
        upstream_checksum = _validate_filter_source_lineage(
            artifact=artifact,
            manifest=manifest,
            marker=marker,
            visited_artifact_ids=visited_artifact_ids,
        )
    return _content_checksum(
        {
            "kind": "resolution" if resolution is not None else "filter_rerun",
            "marker": marker,
            "upstream_lineage_checksum": upstream_checksum,
        }
    )


def _validate_filter_source_lineage(
    *,
    artifact: object,
    manifest: tuple[object, ...],
    marker: dict[str, object],
    visited_artifact_ids: frozenset[int],
) -> str:
    """Lock and authenticate the immutable Task 9 source of a filter rerun."""

    from apps.knowledge_graph.models import (
        CollectionArtifactInput,
        CollectionEntity,
        CollectionEntityDocumentLink,
        GraphArtifact,
        GraphBuildRun,
    )

    source_artifact_id = marker.get("source_artifact_id")
    if type(source_artifact_id) is not int or source_artifact_id < 1:
        raise CollectionGraphAssemblyError(
            "filter rerun source artifact identity is invalid"
        )
    try:
        source = GraphArtifact.objects.select_for_update().get(pk=source_artifact_id)
    except GraphArtifact.DoesNotExist as exc:
        raise CollectionGraphAssemblyError(
            "filter rerun source artifact is missing"
        ) from exc
    if (
        source.scope_type != GraphArtifact.ScopeType.COLLECTION
        or source.scope_id != artifact.scope_id
        or source.status
        not in {GraphArtifact.Status.ACTIVE, GraphArtifact.Status.SUPERSEDED}
        or source.source_hash != marker["source_hash"]
        or source.source_hash != artifact.source_hash
        or source.ontology_version != artifact.ontology_version
        or source.ontology_checksum != artifact.ontology_checksum
        or source.extractor_version != artifact.extractor_version
        or source.resolver_version != artifact.resolver_version
        or source.embedding_model_signature != artifact.embedding_model_signature
        or source.resolution_config_checksum
        != artifact.resolution_config_checksum
        or source.assembly_version != artifact.assembly_version
        or source.assembly_config_checksum != artifact.assembly_config_checksum
    ):
        raise CollectionGraphAssemblyError(
            "filter rerun source artifact identity does not match the destination"
        )
    source_manifest = tuple(
        CollectionArtifactInput.objects.select_for_update()
        .select_related("document_artifact", "collection")
        .filter(artifact=source)
        .order_by("document_artifact_id")
    )

    def manifest_identity(row: object) -> tuple[object, ...]:
        return (
            row.document_artifact_id,
            row.document_id,
            row.membership_signature,
            row.source_signature,
        )

    if tuple(map(manifest_identity, source_manifest)) != tuple(
        map(manifest_identity, manifest)
    ):
        raise CollectionGraphAssemblyError(
            "filter rerun source manifest differs from the destination snapshot"
        )
    try:
        source_run = GraphBuildRun.objects.select_for_update().get(
            pk=marker["source_build_run_id"],
            artifact=source,
        )
    except GraphBuildRun.DoesNotExist as exc:
        raise CollectionGraphAssemblyError(
            "filter rerun source build run is missing"
        ) from exc
    source_stats = source_run.stats if type(source_run.stats) is dict else {}
    resolution_marker = source_stats.get("collection_resolution_commit")
    filter_marker = source_stats.get("filter_commit")
    if (resolution_marker is None) == (filter_marker is None):
        raise CollectionGraphAssemblyError(
            "filter rerun source build run has no exact Task 9 commit"
        )
    source_task9_marker = (
        resolution_marker if resolution_marker is not None else filter_marker
    )
    assembly_marker = source_stats.get("collection_assembly_commit")
    if assembly_marker is None:
        assembly_marker_checksum = None
    else:
        assembly_marker = _require_marker_hash(
            assembly_marker, "filter source collection assembly"
        )
        assembly_marker_checksum = assembly_marker["marker_checksum"]
        source_metadata = source.metadata if type(source.metadata) is dict else {}
        if source_metadata.get("assembly") != {
            "version": source.assembly_version,
            "config_checksum": source.assembly_config_checksum,
            "marker_checksum": assembly_marker_checksum,
        }:
            raise CollectionGraphAssemblyError(
                "filter rerun source assembly audit is invalid"
            )
    if (
        _content_checksum(source_task9_marker)
        != marker["source_task9_marker_checksum"]
        or assembly_marker_checksum != marker["source_assembly_marker_checksum"]
    ):
        raise CollectionGraphAssemblyError(
            "filter rerun source commit checksum is invalid"
        )
    source_entities = tuple(
        CollectionEntity.objects.select_for_update()
        .filter(artifact=source)
        .order_by("pk")
    )
    source_links = tuple(
        CollectionEntityDocumentLink.objects.select_for_update()
        .select_related("collection_entity", "document_entity", "manifest_input")
        .filter(artifact=source)
        .order_by("pk")
    )
    return _validate_task9_lineage(
        source,
        source_run,
        source_manifest,
        source_entities,
        source_links,
        visited_artifact_ids=visited_artifact_ids,
    )


def _lock_current_contributors(collection: object):
    from apps.documents.models import DESCENDED_FROM_DOCUMENT
    from apps.knowledge_graph.models import GraphArtifact

    document_models = tuple(
        sorted(DESCENDED_FROM_DOCUMENT, key=lambda value: value._meta.label)
    )
    document_references = tuple(
        row
        for model in document_models
        for row in model.objects.filter(collection_id=collection.pk).order_by("pk")
    )
    document_ids = [str(row.id) for row in document_references]
    if len(document_ids) != len(set(document_ids)):
        raise CollectionGraphAssemblyError(
            "collection contains duplicate concrete document identities"
        )
    reference_by_id = {str(row.id): row for row in document_references}
    # Global contributor lock order matches Task 9 snapshot creation:
    # source GraphArtifact PKs first, then their concrete Document rows.
    active = tuple(
        GraphArtifact.objects.select_for_update()
        .filter(
            scope_type=GraphArtifact.ScopeType.DOCUMENT,
            scope_id__in=document_ids,
            status=GraphArtifact.Status.ACTIVE,
        )
        .order_by("pk")
    )
    if len(active) != len({row.scope_id for row in active}):
        raise CollectionGraphAssemblyError(
            "collection has multiple active artifacts for one document"
        )
    documents: list[object] = []
    for source in active:
        reference = reference_by_id.get(source.scope_id)
        if reference is None:
            raise CollectionGraphAssemblyError(
                "active document artifact escaped collection membership"
        )
        model = type(reference)
        document = model.objects.select_for_update().get(pk=reference.pk)
        if (
            str(document.id) != source.scope_id
            or document.collection_id != collection.pk
        ):
            raise CollectionGraphAssemblyError(
                "contributing document moved during collection locking"
            )
        documents.append(document)
    current_document_ids = tuple(
        sorted(
            str(document_id)
            for model in document_models
            for document_id in model.objects.filter(
                collection_id=collection.pk
            ).values_list("id", flat=True)
        )
    )
    if current_document_ids != tuple(sorted(document_ids)):
        raise CollectionGraphAssemblyError(
            "collection membership changed during contributor locking"
        )
    current_active_ids = tuple(
        GraphArtifact.objects.filter(
            scope_type=GraphArtifact.ScopeType.DOCUMENT,
            scope_id__in=current_document_ids,
            status=GraphArtifact.Status.ACTIVE,
        )
        .order_by("pk")
        .values_list("pk", flat=True)
    )
    if current_active_ids != tuple(row.pk for row in active):
        raise CollectionGraphAssemblyError(
            "active document artifact snapshot changed during contributor locking"
        )
    return tuple(documents), active


def _validate_locked_manifest(
    collection: object,
    artifact: object,
    manifest: tuple[object, ...],
    aggregate_source_signature: str,
) -> tuple[object, ...]:
    from apps.knowledge_graph.resolution.collection import (
        _snapshot_from_locked_manifest,
    )

    _documents, active_sources = _lock_current_contributors(collection)
    manifest_source_ids = tuple(sorted(row.document_artifact_id for row in manifest))
    current_source_ids = tuple(sorted(row.pk for row in active_sources))
    if manifest_source_ids != current_source_ids:
        raise CollectionGraphAssemblyError(
            "collection active document artifact snapshot changed"
        )
    try:
        snapshot = _snapshot_from_locked_manifest(artifact, manifest)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise CollectionGraphAssemblyError(
            "collection manifest is stale or corrupt"
        ) from exc
    if (
        snapshot.source_hash != aggregate_source_signature
        or artifact.source_hash != aggregate_source_signature
    ):
        raise CollectionGraphAssemblyError(
            "aggregate source signature does not match the exact manifest"
        )
    return active_sources


def _load_locked_task9_rows(artifact: object, config: AssemblyConfig):
    from apps.knowledge_graph.models import (
        CollectionEntity,
        CollectionEntityDocumentLink,
    )

    entities = tuple(
        CollectionEntity.objects.select_for_update()
        .filter(artifact=artifact)
        .order_by("pk")
    )
    if len(entities) > config.max_entities:
        raise CollectionGraphAssemblyError("assembly entity cap exceeded")
    link_query = (
        CollectionEntityDocumentLink.objects.select_for_update()
        .select_related(
            "collection_entity",
            "document_entity",
            "manifest_input",
        )
        .filter(artifact=artifact)
        .order_by("pk")
    )
    if link_query.count() > config.max_links:
        raise CollectionGraphAssemblyError("assembly link cap exceeded")
    links = tuple(link_query)
    return entities, links


def _load_assembly_evidence(
    artifact: object,
    manifest: tuple[object, ...],
    entities: tuple[object, ...],
    links: tuple[object, ...],
    config: AssemblyConfig,
):
    from apps.knowledge_graph.models import (
        CollectionEntityDocumentLink,
        DocumentEntityMention,
        RelationMention,
    )

    source_artifact_ids = {row.document_artifact_id for row in manifest}
    manifest_by_artifact = {row.document_artifact_id: row for row in manifest}
    if len(manifest_by_artifact) != len(manifest):
        raise CollectionGraphAssemblyError("collection manifest source duplicates")
    relation_query = (
        RelationMention.objects.select_for_update()
        .select_related("head", "tail", "chunk", "artifact")
        .filter(artifact_id__in=source_artifact_ids)
        .order_by("pk")
    )
    if relation_query.count() > config.max_evidence:
        raise CollectionGraphAssemblyError("assembly evidence cap exceeded")
    relation_mentions = tuple(relation_query)
    endpoint_ids = {
        endpoint_id
        for row in relation_mentions
        for endpoint_id in (row.head_id, row.tail_id)
    }
    membership_rows = []
    for endpoint_batch in _id_batches(endpoint_ids):
        membership_rows.extend(
            DocumentEntityMention.objects.select_for_update()
            .select_related("document_entity", "mention")
            .filter(
                mention_id__in=endpoint_batch,
                status=DocumentEntityMention.Status.ACTIVE,
            )
            .order_by("mention_id")
        )
    memberships = tuple(membership_rows)
    membership_by_mention = {row.mention_id: row for row in memberships}
    if len(membership_by_mention) != len(memberships):
        raise CollectionGraphAssemblyError(
            "relation evidence has ambiguous active Task 8 mention assignments"
        )
    source_entity_ids = {row.document_entity_id for row in memberships}
    automatic_links = tuple(
        row
        for row in links
        if row.document_entity_id in source_entity_ids
        and row.outcome == CollectionEntityDocumentLink.Outcome.AUTOMATIC
        and row.status == CollectionEntityDocumentLink.Status.ACTIVE
    )
    automatic_by_source = {row.document_entity_id: row for row in automatic_links}
    if len(automatic_by_source) != len(automatic_links):
        raise CollectionGraphAssemblyError(
            "relation evidence has ambiguous automatic Task 9 mappings"
        )
    mapping_by_id = {row.pk: row for row in automatic_links}
    entity_by_id = {row.pk: row for row in entities}
    inputs: list[AssemblyEvidenceInput] = []
    for mention in relation_mentions:
        manifest_input = manifest_by_artifact.get(mention.artifact_id)
        if (
            manifest_input is None
            or mention.artifact.status != mention.artifact.Status.ACTIVE
            or mention.chunk_id is None
            or mention.document_id != manifest_input.document_id
            or mention.chunk.doc_id != manifest_input.document_id
            or mention.head.artifact_id != mention.artifact_id
            or mention.tail.artifact_id != mention.artifact_id
            or mention.head.document_id != mention.document_id
            or mention.tail.document_id != mention.document_id
            or not mention._endpoint_has_relation_chunk_observation(mention.head)
            or not mention._endpoint_has_relation_chunk_observation(mention.tail)
        ):
            raise CollectionGraphAssemblyError(
                "relation evidence escaped the active document manifest"
            )
        head_membership = membership_by_mention.get(mention.head_id)
        tail_membership = membership_by_mention.get(mention.tail_id)
        head_mapping = (
            None
            if head_membership is None
            else automatic_by_source.get(head_membership.document_entity_id)
        )
        tail_mapping = (
            None
            if tail_membership is None
            else automatic_by_source.get(tail_membership.document_entity_id)
        )
        for membership, mapping in (
            (head_membership, head_mapping),
            (tail_membership, tail_mapping),
        ):
            if membership is None:
                continue
            if (
                membership.document_entity.status
                != membership.document_entity.Status.ACTIVE
                or membership.document_entity.artifact_id != mention.artifact_id
                or membership.document_entity.document_id != mention.document_id
            ):
                raise CollectionGraphAssemblyError(
                    "relation evidence violates the exact Task 8 assignment"
                )
            if mapping is not None and (
                mapping.artifact_id != artifact.pk
                or mapping.manifest_input_id != manifest_input.pk
                or mapping.document_entity_id != membership.document_entity_id
                or mapping.resolver_version != artifact.resolver_version
            ):
                raise CollectionGraphAssemblyError(
                    "relation evidence mapping violates the exact Task 9 assignment"
                )
        head_entity = (
            None
            if head_mapping is None
            else entity_by_id.get(head_mapping.collection_entity_id)
        )
        tail_entity = (
            None
            if tail_mapping is None
            else entity_by_id.get(tail_mapping.collection_entity_id)
        )
        if (head_mapping is not None and head_entity is None) or (
            tail_mapping is not None and tail_entity is None
        ):
            raise CollectionGraphAssemblyError(
                "relation mapping endpoint escaped the destination artifact"
            )
        inputs.append(
            AssemblyEvidenceInput(
                relation_mention_id=mention.pk,
                document_artifact_id=mention.artifact_id,
                chunk_id=mention.chunk_id,
                head_mention_id=mention.head_id,
                tail_mention_id=mention.tail_id,
                head_mapping_id=(None if head_mapping is None else head_mapping.pk),
                tail_mapping_id=(None if tail_mapping is None else tail_mapping.pk),
                head_collection_entity_id=(
                    None if head_entity is None else head_entity.pk
                ),
                tail_collection_entity_id=(
                    None if tail_entity is None else tail_entity.pk
                ),
                head_entity_type=(
                    None if head_entity is None else head_entity.entity_type
                ),
                tail_entity_type=(
                    None if tail_entity is None else tail_entity.entity_type
                ),
                head_status=None if head_entity is None else head_entity.status,
                tail_status=None if tail_entity is None else tail_entity.status,
                relation_type=mention.relation_type,
                extraction_confidence=mention.extraction_confidence,
                head_cluster_key=(
                    None if head_entity is None else head_entity.cluster_key
                ),
                tail_cluster_key=(
                    None if tail_entity is None else tail_entity.cluster_key
                ),
            )
        )
    return tuple(inputs), relation_mentions, mapping_by_id


def _active_entity_provenance(
    entities: tuple[object, ...],
    links: tuple[object, ...],
) -> tuple[frozenset[int], frozenset[int]]:
    from apps.knowledge_graph.models import (
        CollectionEntityDocumentLink,
        DocumentEntityMention,
    )

    active_ids = frozenset(
        row.pk for row in entities if row.status == row.Status.ACTIVE
    )
    automatic = tuple(
        row
        for row in links
        if row.outcome == CollectionEntityDocumentLink.Outcome.AUTOMATIC
        and row.status == CollectionEntityDocumentLink.Status.ACTIVE
        and row.collection_entity_id in active_ids
    )
    if any(
        row.document_entity.status != row.document_entity.Status.ACTIVE
        or row.document_entity.artifact_id != row.manifest_input.document_artifact_id
        or row.document_entity.document_id != row.manifest_input.document_id
        or row.collection_entity.status != row.collection_entity.Status.ACTIVE
        for row in automatic
    ):
        raise CollectionGraphAssemblyError(
            "active collection entity provenance mapping is invalid"
        )
    document_entity_ids = {row.document_entity_id for row in automatic}
    provenanced_documents = set()
    for document_entity_batch in _id_batches(document_entity_ids):
        memberships = (
            DocumentEntityMention.objects.select_for_update()
            .select_related("mention", "document_entity")
            .filter(
                document_entity_id__in=document_entity_batch,
                status=DocumentEntityMention.Status.ACTIVE,
            )
            .order_by("pk")
        )
        for row in memberships.iterator(chunk_size=_ASSEMBLY_INSERT_BATCH_SIZE):
            if (
                row.mention_id
                and row.mention.chunk_id
                and row.mention.artifact_id == row.document_entity.artifact_id
                and row.mention.document_id == row.document_entity.document_id
            ):
                provenanced_documents.add(row.document_entity_id)
    provenanced_entities = frozenset(
        row.collection_entity_id
        for row in automatic
        if row.document_entity_id in provenanced_documents
    )
    return active_ids, provenanced_entities


def _projection_stats_content(stats: AssemblyProjectionStats) -> dict[str, object]:
    return {
        "entity_count": stats.entity_count,
        "relation_count": stats.relation_count,
        "evidence_count": stats.evidence_count,
        "promoted_evidence_count": stats.promoted_evidence_count,
        "suppressed_evidence_count": stats.suppressed_evidence_count,
        "rejected_evidence_count": stats.rejected_evidence_count,
        "orphan_count": stats.orphan_count,
        "orphan_ratio": stats.orphan_ratio,
    }


def _ordered_checksum_root(checksums: tuple[str, ...], namespace: str) -> str:
    if type(namespace) is not str or not namespace:
        raise CollectionGraphAssemblyError("checksum root namespace is invalid")
    digest = sha256(f"{namespace}\0v1\0{len(checksums)}\0".encode())
    for checksum in checksums:
        if type(checksum) is not str or not _HASH_PATTERN.fullmatch(checksum):
            raise CollectionGraphAssemblyError("row checksum root input is invalid")
        digest.update(bytes.fromhex(checksum))
    return digest.hexdigest()


def _assembly_marker(
    *,
    artifact: object,
    plan: CollectionAssemblyPlan,
    projection_stats: AssemblyProjectionStats,
    config: AssemblyConfig,
    lineage_checksum: str,
    relation_checksums: tuple[str, ...],
    evidence_checksums: tuple[str, ...],
) -> dict[str, object]:
    config_checksum = assembly_config_checksum(config)
    if (
        config.version != artifact.assembly_version
        or config_checksum != artifact.assembly_config_checksum
    ):
        raise CollectionGraphAssemblyError(
            "assembly marker config differs from immutable artifact identity"
        )
    marker: dict[str, object] = {
        "version": 1,
        "assembly_version": artifact.assembly_version,
        "config": _config_payload(config),
        "assembly_config_checksum": artifact.assembly_config_checksum,
        "source_hash": artifact.source_hash,
        "ontology_version": artifact.ontology_version,
        "ontology_checksum": artifact.ontology_checksum,
        "extractor_version": artifact.extractor_version,
        "resolver_version": artifact.resolver_version,
        "filter_policy_version": artifact.filter_policy_version,
        "filter_policy_checksum": artifact.filter_policy_checksum,
        "resolution_config_checksum": artifact.resolution_config_checksum,
        "embedding_model_signature": artifact.embedding_model_signature,
        "task9_lineage_checksum": lineage_checksum,
        "plan_checksum": plan.checksum,
        "projection": _projection_stats_content(projection_stats),
        "relation_row_count": len(relation_checksums),
        "relation_row_checksum_root": _ordered_checksum_root(
            relation_checksums, "collection-relation-rows"
        ),
        "evidence_row_count": len(evidence_checksums),
        "evidence_row_checksum_root": _ordered_checksum_root(
            evidence_checksums, "collection-evidence-rows"
        ),
    }
    marker["marker_checksum"] = _marker_hash(marker)
    return marker


def _validate_existing_assembly(
    *,
    artifact: object,
    run: object,
    plan: CollectionAssemblyPlan,
    projection_stats: AssemblyProjectionStats,
    config: AssemblyConfig,
    lineage_checksum: str,
    relations: tuple[object, ...],
    evidence_rows: tuple[object, ...],
) -> AssemblyResult | None:
    stats = run.stats if type(run.stats) is dict else {}
    marker = stats.get("collection_assembly_commit")
    if marker is None:
        if relations or evidence_rows:
            raise CollectionGraphAssemblyError(
                "collection relation rows exist without an assembly commit marker"
            )
        return None
    marker = _require_marker_hash(marker, "collection assembly")
    relation_by_key = {
        (row.source_id, row.relation_type, row.target_id): row for row in relations
    }
    if len(relation_by_key) != len(relations):
        raise CollectionGraphAssemblyError("persisted collection relations duplicate")
    planned_by_key = {row.key: row for row in plan.relations}
    if relation_by_key.keys() != planned_by_key.keys():
        raise CollectionGraphAssemblyError(
            "persisted collection relations differ from the locked projection"
        )
    for key, row in relation_by_key.items():
        expected = planned_by_key[key]
        metadata = row.metadata if type(row.metadata) is dict else {}
        if (
            row.status != row.Status.ACTIVE
            or row.support_count != expected.support_count
            or row.confidence != expected.confidence
            or metadata.get("row_audit_checksum") != _relation_row_audit(row)
        ):
            raise CollectionGraphAssemblyError(
                "persisted collection relation audit is corrupt"
            )
    evidence_by_mention = {row.relation_mention_id: row for row in evidence_rows}
    if len(evidence_by_mention) != len(evidence_rows):
        raise CollectionGraphAssemblyError("persisted relation evidence duplicates")
    planned_evidence = {row.relation_mention_id: row for row in plan.evidence}
    if evidence_by_mention.keys() != planned_evidence.keys():
        raise CollectionGraphAssemblyError(
            "persisted evidence differs from the locked projection"
        )
    status_by_disposition = {
        EvidenceDisposition.PROMOTED: "active",
        EvidenceDisposition.SUPPRESSED: "suppressed",
        EvidenceDisposition.REJECTED: "rejected",
    }
    support_by_relation: dict[int, int] = {}
    for mention_id, row in evidence_by_mention.items():
        expected = planned_evidence[mention_id]
        expected_relation = (
            None
            if expected.relation_key is None
            else relation_by_key[expected.relation_key].pk
        )
        metadata = row.metadata if type(row.metadata) is dict else {}
        if (
            row.artifact_id != artifact.pk
            or row.relation_id != expected_relation
            or row.head_mapping_id != expected.head_mapping_id
            or row.tail_mapping_id != expected.tail_mapping_id
            or row.status != status_by_disposition[expected.disposition]
            or row.reason != expected.reason
            or row.orientation != expected.orientation
            or row.ontology_checksum != artifact.ontology_checksum
            or row.assembly_config_checksum != assembly_config_checksum(config)
            or metadata.get("row_audit_checksum") != _evidence_row_audit(row)
        ):
            raise CollectionGraphAssemblyError(
                "persisted collection evidence audit is corrupt"
            )
        if row.relation_id is not None:
            support_by_relation[row.relation_id] = (
                support_by_relation.get(row.relation_id, 0) + 1
            )
    if any(
        support_by_relation.get(row.pk, 0) != row.support_count for row in relations
    ):
        raise CollectionGraphAssemblyError(
            "active relation support count differs from real evidence"
        )
    relation_checksums = tuple(
        row.metadata["row_audit_checksum"]
        for row in sorted(relations, key=lambda value: value.pk)
    )
    evidence_checksums = tuple(
        row.metadata["row_audit_checksum"]
        for row in sorted(evidence_rows, key=lambda value: value.relation_mention_id)
    )
    expected_marker = _assembly_marker(
        artifact=artifact,
        plan=plan,
        projection_stats=projection_stats,
        config=config,
        lineage_checksum=lineage_checksum,
        relation_checksums=relation_checksums,
        evidence_checksums=evidence_checksums,
    )
    if marker != expected_marker:
        raise CollectionGraphAssemblyError(
            "assembly marker differs from locked rows or typed build identity"
        )
    artifact_metadata = artifact.metadata if type(artifact.metadata) is dict else {}
    if artifact_metadata.get("assembly") != {
        "version": config.version,
        "config_checksum": assembly_config_checksum(config),
        "marker_checksum": marker["marker_checksum"],
    }:
        raise CollectionGraphAssemblyError(
            "artifact assembly identity is missing or corrupt"
        )
    return AssemblyResult(
        artifact_id=artifact.pk,
        build_run_id=run.pk,
        source_hash=artifact.source_hash,
        plan_checksum=plan.checksum,
        marker_checksum=marker["marker_checksum"],
        stats=projection_stats,
    )


def _write_assembly(
    *,
    artifact: object,
    run: object,
    plan: CollectionAssemblyPlan,
    projection_stats: AssemblyProjectionStats,
    config: AssemblyConfig,
    lineage_checksum: str,
    relation_mentions: tuple[object, ...],
    mapping_by_id: dict[int, object],
) -> AssemblyResult:
    from apps.knowledge_graph.models import (
        CollectionEntity,
        CollectionRelation,
        CollectionRelationEvidence,
    )

    entity_by_id = {
        row.pk: row
        for row in CollectionEntity.objects.select_for_update()
        .filter(artifact=artifact)
        .order_by("pk")
    }
    relation_rows = []
    for planned in plan.relations:
        row = CollectionRelation(
            artifact=artifact,
            source=entity_by_id[planned.source_entity_id],
            relation_type=planned.relation_type,
            target=entity_by_id[planned.target_entity_id],
            status=CollectionRelation.Status.ACTIVE,
            support_count=planned.support_count,
            confidence=planned.confidence,
            metadata={
                "assembly_version": config.version,
                "assembly_config_checksum": assembly_config_checksum(config),
                "plan_checksum": plan.checksum,
                "evidence_mention_ids": list(planned.evidence_mention_ids),
            },
        )
        row.metadata = {
            **row.metadata,
            "row_audit_checksum": _relation_row_audit(row),
        }
        relation_rows.append(row)
    # The locked projection has already validated every cross-row invariant.
    # Use the plain base manager so model full_clean() does not issue O(rows)
    # constraint and endpoint-membership queries before batched inserts.
    CollectionRelation._base_manager.bulk_create(
        relation_rows,
        batch_size=_ASSEMBLY_INSERT_BATCH_SIZE,
    )
    relation_by_key = {row_key.key: row for row_key, row in zip(
        plan.relations, relation_rows, strict=True
    )}
    mention_by_id = {row.pk: row for row in relation_mentions}
    status_by_disposition = {
        EvidenceDisposition.PROMOTED: CollectionRelationEvidence.Status.ACTIVE,
        EvidenceDisposition.SUPPRESSED: CollectionRelationEvidence.Status.SUPPRESSED,
        EvidenceDisposition.REJECTED: CollectionRelationEvidence.Status.REJECTED,
    }
    evidence_rows = []
    config_checksum = assembly_config_checksum(config)
    for planned in plan.evidence:
        mention = mention_by_id[planned.relation_mention_id]
        row = CollectionRelationEvidence(
            artifact=artifact,
            relation=(
                None
                if planned.relation_key is None
                else relation_by_key[planned.relation_key]
            ),
            relation_mention=mention,
            head_mapping=(
                None
                if planned.head_mapping_id is None
                else mapping_by_id[planned.head_mapping_id]
            ),
            tail_mapping=(
                None
                if planned.tail_mapping_id is None
                else mapping_by_id[planned.tail_mapping_id]
            ),
            status=status_by_disposition[planned.disposition],
            reason=planned.reason,
            orientation=planned.orientation,
            ontology_checksum=artifact.ontology_checksum,
            assembly_config_checksum=config_checksum,
            metadata={
                "assembly_version": config.version,
                "plan_checksum": plan.checksum,
                "disposition": planned.disposition.value,
            },
        )
        row.metadata = {
            **row.metadata,
            "row_audit_checksum": _evidence_row_audit(row),
        }
        evidence_rows.append(row)
    CollectionRelationEvidence._base_manager.bulk_create(
        evidence_rows,
        batch_size=_ASSEMBLY_INSERT_BATCH_SIZE,
    )
    relation_checksums = tuple(
        row.metadata["row_audit_checksum"]
        for row in sorted(relation_rows, key=lambda value: value.pk)
    )
    evidence_checksums = tuple(
        row.metadata["row_audit_checksum"]
        for row in sorted(evidence_rows, key=lambda value: value.relation_mention_id)
    )
    marker = _assembly_marker(
        artifact=artifact,
        plan=plan,
        projection_stats=projection_stats,
        config=config,
        lineage_checksum=lineage_checksum,
        relation_checksums=relation_checksums,
        evidence_checksums=evidence_checksums,
    )
    stats = run.stats if type(run.stats) is dict else {}
    run.stats = {**stats, "collection_assembly_commit": marker}
    run.stage = run.Stage.PERSISTENCE
    run.status = run.Status.RUNNING
    run.save(update_fields=["stats", "stage", "status"])
    metadata = artifact.metadata if type(artifact.metadata) is dict else {}
    artifact.metadata = {
        **metadata,
        "assembly": {
            "version": config.version,
            "config_checksum": config_checksum,
            "marker_checksum": marker["marker_checksum"],
        },
    }
    artifact.save(update_fields=["metadata"])
    return AssemblyResult(
        artifact_id=artifact.pk,
        build_run_id=run.pk,
        source_hash=artifact.source_hash,
        plan_checksum=plan.checksum,
        marker_checksum=marker["marker_checksum"],
        stats=projection_stats,
    )


def _locked_projection(
    *,
    collection: object,
    artifact: object,
    run: object,
    manifest: tuple[object, ...],
    aggregate_source_signature: str,
    ontology: object,
    config: AssemblyConfig,
):
    _validate_locked_manifest(
        collection, artifact, manifest, aggregate_source_signature
    )
    entities, links = _load_locked_task9_rows(artifact, config)
    lineage_checksum = _validate_task9_lineage(
        artifact, run, manifest, entities, links
    )
    (
        evidence_inputs,
        relation_mentions,
        mapping_by_id,
    ) = _load_assembly_evidence(artifact, manifest, entities, links, config)
    try:
        plan = plan_collection_relations(evidence_inputs, ontology, config)
    except (TypeError, ValueError) as exc:
        raise CollectionGraphAssemblyError(
            "collection relation projection failed deterministic validation"
        ) from exc
    active_ids, provenanced_ids = _active_entity_provenance(entities, links)
    try:
        projection_stats = validate_assembly_projection(
            plan,
            active_entity_ids=active_ids,
            provenanced_entity_ids=provenanced_ids,
            config=config,
        )
    except ValueError as exc:
        raise CollectionGraphAssemblyError(str(exc)) from exc
    return (
        plan,
        projection_stats,
        lineage_checksum,
        relation_mentions,
        mapping_by_id,
    )


def _candidate_identity(artifact: object, run: object, collection_id: int) -> None:
    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun

    if (
        artifact.scope_type != GraphArtifact.ScopeType.COLLECTION
        or artifact.scope_id != str(collection_id)
        or run.artifact_id != artifact.pk
        or run.scope_type != GraphArtifact.ScopeType.COLLECTION
        or run.scope_id != str(collection_id)
        or run.build_kind != GraphBuildRun.BuildKind.COLLECTION
    ):
        raise CollectionGraphAssemblyError(
            "build run and artifact do not identify the collection"
        )
    for field_name in (
        "source_hash",
        "ontology_version",
        "extractor_version",
        "resolver_version",
        "filter_policy_version",
        "embedding_model_signature",
        "ontology_checksum",
        "filter_policy_checksum",
        "resolution_config_checksum",
        "assembly_version",
        "assembly_config_checksum",
    ):
        if getattr(run, field_name) != getattr(artifact, field_name):
            raise CollectionGraphAssemblyError(
                f"build run {field_name} differs from candidate artifact"
            )
    if artifact.status not in {
        GraphArtifact.Status.BUILDING,
        GraphArtifact.Status.ACTIVE,
    }:
        raise CollectionGraphAssemblyError(
            "collection assembly candidate must be building or already active"
        )


def _locked_candidate(
    collection_id: int,
    build_run_id: int,
    *,
    lock_competing_runs: bool = False,
):
    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun

    collection = lock_collection_graph_scope(collection_id)
    run_reference = GraphBuildRun.objects.only("artifact_id").get(pk=build_run_id)
    scope_artifacts = tuple(
        GraphArtifact.objects.select_for_update()
        .filter(
            scope_type=GraphArtifact.ScopeType.COLLECTION,
            scope_id=str(collection_id),
        )
        .order_by("pk")
    )
    artifact_by_id = {row.pk: row for row in scope_artifacts}
    artifact = artifact_by_id.get(run_reference.artifact_id)
    if artifact is None:
        raise CollectionGraphAssemblyError(
            "build run artifact is outside the locked collection scope"
        )
    if lock_competing_runs:
        locked_runs = tuple(
            GraphBuildRun.objects.select_for_update()
            .filter(artifact_id__in=artifact_by_id)
            .order_by("pk")
        )
        run = next((row for row in locked_runs if row.pk == build_run_id), None)
        if run is None:
            raise CollectionGraphAssemblyError(
                "candidate build run changed before scope locking"
            )
    else:
        run = GraphBuildRun.objects.select_for_update().get(pk=build_run_id)
    _candidate_identity(artifact, run, collection_id)
    return collection, artifact, run, scope_artifacts


def assemble_collection_graph(
    collection_id: int,
    build_run_id: int,
    aggregate_source_signature: str,
    *,
    ontology: object | None = None,
    config: AssemblyConfig | None = None,
) -> AssemblyResult:
    """Commit relations/evidence to the existing Task 9 shadow artifact only."""

    from django.db import transaction

    from apps.knowledge_graph.models import (
        CollectionArtifactInput,
        CollectionRelation,
        CollectionRelationEvidence,
        GraphArtifact,
    )

    collection_id = _positive_int(collection_id, "collection id")
    build_run_id = _positive_int(build_run_id, "build run id")
    aggregate_source_signature = _source_hash(aggregate_source_signature)
    with transaction.atomic():
        collection, artifact, run, _scope_artifacts = _locked_candidate(
            collection_id, build_run_id
        )
        config = _resolve_config(run, config)
        ontology = _resolve_ontology(artifact, ontology)
        manifest = tuple(
            CollectionArtifactInput.objects.select_for_update()
            .select_related("document_artifact", "collection")
            .filter(artifact=artifact)
            .order_by("document_artifact_id")
        )
        (
            plan,
            projection_stats,
            lineage_checksum,
            relation_mentions,
            mapping_by_id,
        ) = _locked_projection(
            collection=collection,
            artifact=artifact,
            run=run,
            manifest=manifest,
            aggregate_source_signature=aggregate_source_signature,
            ontology=ontology,
            config=config,
        )
        relations = tuple(
            CollectionRelation.objects.select_for_update()
            .filter(artifact=artifact)
            .order_by("pk")
        )
        evidence_rows = tuple(
            CollectionRelationEvidence.objects.select_for_update()
            .filter(artifact=artifact)
            .order_by("relation_mention_id")
        )
        existing = _validate_existing_assembly(
            artifact=artifact,
            run=run,
            plan=plan,
            projection_stats=projection_stats,
            config=config,
            lineage_checksum=lineage_checksum,
            relations=relations,
            evidence_rows=evidence_rows,
        )
        if existing is not None:
            return existing
        if artifact.status != GraphArtifact.Status.BUILDING:
            raise CollectionGraphAssemblyError(
                "active artifact has no valid assembly commit"
            )
        return _write_assembly(
            artifact=artifact,
            run=run,
            plan=plan,
            projection_stats=projection_stats,
            config=config,
            lineage_checksum=lineage_checksum,
            relation_mentions=relation_mentions,
            mapping_by_id=mapping_by_id,
        )


_COLLECTION_ACTIVATION_LOCK_BASE = 5_497_230_000_000_000_000


def _lock_collection_scope(cursor: object, collection_id: int) -> None:
    """Serialize swaps even when a collection has no active artifact yet."""

    cursor.execute(
        "SELECT pg_advisory_xact_lock(%s)",
        [_COLLECTION_ACTIVATION_LOCK_BASE + collection_id],
    )


def lock_collection_graph_scope(collection_id: int):
    """Enter the one global lock order for collection graph operations."""

    from django.db import connection

    from apps.collections.models import Collection

    collection_id = _positive_int(collection_id, "collection id")
    with connection.cursor() as cursor:
        _lock_collection_scope(cursor, collection_id)
    return Collection.objects.select_for_update().get(pk=collection_id)


def _newer_activation_exists(
    candidate_artifact_id: int,
    scope_artifacts: tuple[object, ...],
) -> bool:
    """Use the durable activation timestamp as the monotonic version fence."""

    return any(
        row.pk > candidate_artifact_id
        and (row.activated_at is not None or row.status == "active")
        for row in scope_artifacts
    )


def _validate_locked_complete_artifact(
    *,
    collection: object,
    artifact: object,
    run: object,
    aggregate_source_signature: str,
    ontology: object | None,
    config: AssemblyConfig | None,
) -> AssemblyResult:
    from apps.knowledge_graph.models import (
        CollectionArtifactInput,
        CollectionRelation,
        CollectionRelationEvidence,
    )

    config = _resolve_config(run, config)
    ontology = _resolve_ontology(artifact, ontology)
    manifest = tuple(
        CollectionArtifactInput.objects.select_for_update()
        .select_related("document_artifact", "collection")
        .filter(artifact=artifact)
        .order_by("document_artifact_id")
    )
    (
        plan,
        projection_stats,
        lineage_checksum,
        _relation_mentions,
        _mapping_by_id,
    ) = _locked_projection(
        collection=collection,
        artifact=artifact,
        run=run,
        manifest=manifest,
        aggregate_source_signature=aggregate_source_signature,
        ontology=ontology,
        config=config,
    )
    relations = tuple(
        CollectionRelation.objects.select_for_update()
        .filter(artifact=artifact)
        .order_by("pk")
    )
    evidence_rows = tuple(
        CollectionRelationEvidence.objects.select_for_update()
        .filter(artifact=artifact)
        .order_by("relation_mention_id")
    )
    result = _validate_existing_assembly(
        artifact=artifact,
        run=run,
        plan=plan,
        projection_stats=projection_stats,
        config=config,
        lineage_checksum=lineage_checksum,
        relations=relations,
        evidence_rows=evidence_rows,
    )
    if result is None:
        raise CollectionGraphAssemblyError(
            "candidate has no complete collection assembly commit"
        )
    return result


def validate_collection_graph_artifact(
    collection_id: int,
    build_run_id: int,
    aggregate_source_signature: str,
    *,
    ontology: object | None = None,
    config: AssemblyConfig | None = None,
) -> AssemblyResult:
    """Revalidate a complete shadow without changing current graph state."""

    from django.db import transaction

    collection_id = _positive_int(collection_id, "collection id")
    build_run_id = _positive_int(build_run_id, "build run id")
    aggregate_source_signature = _source_hash(aggregate_source_signature)
    with transaction.atomic():
        collection, artifact, run, _scope_artifacts = _locked_candidate(
            collection_id,
            build_run_id,
            lock_competing_runs=True,
        )
        return _validate_locked_complete_artifact(
            collection=collection,
            artifact=artifact,
            run=run,
            aggregate_source_signature=aggregate_source_signature,
            ontology=ontology,
            config=config,
        )


def activate_collection_graph(
    collection_id: int,
    build_run_id: int,
    aggregate_source_signature: str,
    *,
    ontology: object | None = None,
    config: AssemblyConfig | None = None,
) -> AssemblyResult:
    """Atomically expose a validated shadow while enforcing monotonic newer-wins."""

    from django.db import transaction
    collection_id = _positive_int(collection_id, "collection id")
    build_run_id = _positive_int(build_run_id, "build run id")
    aggregate_source_signature = _source_hash(aggregate_source_signature)
    with transaction.atomic():
        # _locked_candidate performs deterministic select_for_update locking for
        # the advisory scope, Collection, competing artifacts/runs, and candidate.
        collection, artifact, run, scope_artifacts = _locked_candidate(
            collection_id,
            build_run_id,
            lock_competing_runs=True,
        )
        result = _validate_locked_complete_artifact(
            collection=collection,
            artifact=artifact,
            run=run,
            aggregate_source_signature=aggregate_source_signature,
            ontology=ontology,
            config=config,
        )
        _swap_active_collection_artifact(
            artifact=artifact,
            run=run,
            scope_artifacts=scope_artifacts,
        )
        return result


def _swap_active_collection_artifact(
    *,
    artifact: object,
    run: object,
    scope_artifacts: tuple[object, ...],
) -> None:
    """Swap a locked, validated candidate without rewriting prior completion."""

    from django.utils import timezone

    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun

    active = tuple(
        row for row in scope_artifacts if row.status == GraphArtifact.Status.ACTIVE
    )
    if _newer_activation_exists(artifact.pk, scope_artifacts):
        raise CollectionGraphAssemblyError(
            "a newer collection artifact already won activation"
        )
    if artifact.status == GraphArtifact.Status.ACTIVE:
        if tuple(row.pk for row in active) != (artifact.pk,):
            raise CollectionGraphAssemblyError(
                "active candidate is not the unique current collection artifact"
            )
        if (
            run.stage != GraphBuildRun.Stage.COMPLETE
            or run.status != GraphBuildRun.Status.SUCCEEDED
        ):
            run.stage = GraphBuildRun.Stage.COMPLETE
            run.status = GraphBuildRun.Status.SUCCEEDED
            run.finished_at = timezone.now()
            run.save(update_fields=["stage", "status", "finished_at"])
        return
    if len(active) > 1:
        raise CollectionGraphAssemblyError(
            "collection has multiple active artifacts before activation"
        )
    superseded_at = timezone.now()
    for previous in active:
        if previous.superseded_at is not None:
            raise CollectionGraphAssemblyError(
                "active artifact already has a supersession timestamp"
            )
        previous.status = GraphArtifact.Status.SUPERSEDED
        previous.superseded_at = superseded_at
        previous.save(update_fields=["status", "superseded_at"])
    activated_at = timezone.now()
    artifact.status = GraphArtifact.Status.ACTIVE
    artifact.activated_at = activated_at
    artifact.completed_at = activated_at
    artifact.save(update_fields=["status", "activated_at", "completed_at"])
    run.stage = GraphBuildRun.Stage.COMPLETE
    run.status = GraphBuildRun.Status.SUCCEEDED
    run.finished_at = activated_at
    run.save(update_fields=["stage", "status", "finished_at"])


__all__ = [
    "ASSEMBLY_VERSION",
    "ASSEMBLY_V1_MAX_ENTITIES",
    "ASSEMBLY_V1_MAX_EVIDENCE",
    "ASSEMBLY_V1_MAX_LINKS",
    "ASSEMBLY_V1_MAX_ORPHAN_ENTITIES",
    "ASSEMBLY_V1_MAX_RELATIONS",
    "AssemblyConfig",
    "AssemblyEvidenceInput",
    "AssemblyProjectionStats",
    "AssemblyResult",
    "CollectionAssemblyPlan",
    "CollectionGraphAssemblyError",
    "EvidenceDisposition",
    "PlannedEvidence",
    "PlannedRelation",
    "activate_collection_graph",
    "assemble_collection_graph",
    "assembly_config_checksum",
    "lock_collection_graph_scope",
    "plan_collection_relations",
    "validate_assembly_projection",
    "validate_collection_graph_artifact",
]
