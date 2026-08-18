"""Transparent, status-only filtering for collection graph entities."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from math import isfinite, log1p


class FilterStatus(StrEnum):
    ACTIVE = "active"
    SUPPRESSED = "suppressed"
    REJECTED = "rejected"


class PositionKind(StrEnum):
    TITLE = "title"
    ABSTRACT = "abstract"
    CAPTION = "caption"
    BODY = "body"
    OTHER = "other"


def _unit_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number in [0, 1]")
    number = float(value)
    if not isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{label} must be a finite number in [0, 1]")
    return number


def _nonnegative_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be finite and nonnegative")
    number = float(value)
    if not isfinite(number) or number < 0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return number


def _text(value: object, label: str, maximum: int = 256) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be a nonempty exact string")
    normalized = value.strip()
    if len(normalized) > maximum or "\x00" in normalized:
        raise ValueError(f"{label} is invalid or too long")
    return normalized


@dataclass(frozen=True, slots=True)
class UtilityWeights:
    frequency: float = 0.15
    document_dispersion: float = 0.35
    extraction_confidence: float = 0.15
    resolution_confidence: float = 0.15
    relation_participation: float = 0.10
    salient_position: float = 0.10

    def __post_init__(self) -> None:
        values = []
        for name in self.__dataclass_fields__:
            number = _nonnegative_float(getattr(self, name), f"{name} weight")
            object.__setattr__(self, name, number)
            values.append(number)
        if sum(values) <= 0:
            raise ValueError("utility weights must have a positive sum")


@dataclass(frozen=True, slots=True)
class FilterPolicy:
    version: str = "collection-filter-v1"
    frequency_cap: int = 100
    relation_participation_cap: int = 20
    utility_activation_threshold: float = 0.20
    weights: UtilityWeights = UtilityWeights()
    rejected_entity_types: frozenset[str] = frozenset()
    suppressed_entity_types: frozenset[str] = frozenset({"publisher"})

    def __post_init__(self) -> None:
        object.__setattr__(self, "version", _text(self.version, "policy version"))
        for name in ("frequency_cap", "relation_participation_cap"):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= 1_000_000:
                raise ValueError(f"{name} must be a positive bounded integer")
        object.__setattr__(
            self,
            "utility_activation_threshold",
            _unit_float(
                self.utility_activation_threshold, "utility activation threshold"
            ),
        )
        if type(self.weights) is not UtilityWeights:
            raise ValueError("weights must be an exact UtilityWeights value")
        self.weights.__post_init__()
        for name in ("rejected_entity_types", "suppressed_entity_types"):
            values = getattr(self, name)
            if type(values) is not frozenset:
                raise ValueError(f"{name} must be an exact frozenset")
            normalized = frozenset(_text(value, name, 128) for value in values)
            object.__setattr__(self, name, normalized)
        overlap = self.rejected_entity_types.intersection(self.suppressed_entity_types)
        if overlap:
            raise ValueError("entity types cannot be both rejected and suppressed")


@dataclass(frozen=True, slots=True)
class EntityFilterInput:
    entity_id: str
    entity_type: str
    mention_ids: tuple[str, ...]
    document_ids: tuple[str, ...]
    extraction_confidence: float
    resolution_confidence: float
    promotion_confidence: float | None
    relation_participation: int
    positions: tuple[PositionKind, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "entity_id", _text(self.entity_id, "entity id"))
        object.__setattr__(
            self, "entity_type", _text(self.entity_type, "entity type", 128)
        )
        for name in ("mention_ids", "document_ids"):
            values = getattr(self, name)
            if type(values) is not tuple:
                raise ValueError(f"{name} must be an exact tuple")
            normalized = tuple(_text(value, name) for value in values)
            object.__setattr__(self, name, normalized)
        if not self.mention_ids:
            raise ValueError("filter evidence requires at least one raw mention")
        if len(self.mention_ids) != len(set(self.mention_ids)):
            raise ValueError("filter evidence contains duplicate mention IDs")
        if len(self.document_ids) != len(self.mention_ids):
            raise ValueError("every retained mention requires a document ID")
        object.__setattr__(
            self,
            "extraction_confidence",
            _unit_float(self.extraction_confidence, "extraction confidence"),
        )
        object.__setattr__(
            self,
            "resolution_confidence",
            _unit_float(self.resolution_confidence, "resolution confidence"),
        )
        if self.promotion_confidence is not None:
            object.__setattr__(
                self,
                "promotion_confidence",
                _unit_float(self.promotion_confidence, "promotion confidence"),
            )
        if type(self.relation_participation) is not int or not (
            0 <= self.relation_participation <= 2**63 - 1
        ):
            raise ValueError("relation participation must be a nonnegative integer")
        if type(self.positions) is not tuple or any(
            type(position) is not PositionKind for position in self.positions
        ):
            raise ValueError("positions must contain exact PositionKind values")


@dataclass(frozen=True, slots=True)
class EntityFilterDecision:
    entity_id: str
    status: FilterStatus
    reason_codes: tuple[str, ...]
    policy_version: str
    policy_checksum: str
    ontology_checksum: str
    retained_mention_ids: tuple[str, ...]
    extraction_confidence: float
    resolution_confidence: float
    retrieval_utility: float
    promotion_confidence: float | None


def _bounded_log(value: int, cap: int) -> float:
    return log1p(min(value, cap)) / log1p(cap)


def filter_policy_checksum(policy: FilterPolicy) -> str:
    """Return a stable digest covering every status and utility policy input."""

    if type(policy) is not FilterPolicy:
        raise ValueError("policy must be an exact FilterPolicy value")
    policy.__post_init__()
    payload = {
        "version": policy.version,
        "frequency_cap": policy.frequency_cap,
        "relation_participation_cap": policy.relation_participation_cap,
        "utility_activation_threshold": policy.utility_activation_threshold,
        "weights": {
            name: getattr(policy.weights, name)
            for name in policy.weights.__dataclass_fields__
        },
        "rejected_entity_types": sorted(policy.rejected_entity_types),
        "suppressed_entity_types": sorted(policy.suppressed_entity_types),
    }
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def score_retrieval_utility(
    evidence: EntityFilterInput,
    policy: FilterPolicy,
) -> float:
    """Compute an auditable utility score; never overwrite other confidences."""

    if type(evidence) is not EntityFilterInput:
        raise ValueError("evidence must be an exact EntityFilterInput value")
    if type(policy) is not FilterPolicy:
        raise ValueError("policy must be an exact FilterPolicy value")
    evidence.__post_init__()
    policy.__post_init__()
    frequency = len(evidence.mention_ids)
    frequency_score = _bounded_log(frequency, policy.frequency_cap)
    unique_documents = len(set(evidence.document_ids))
    dispersion_score = unique_documents / frequency
    relation_score = _bounded_log(
        evidence.relation_participation,
        policy.relation_participation_cap,
    )
    position_values = {
        PositionKind.TITLE: 1.0,
        PositionKind.ABSTRACT: 0.9,
        PositionKind.CAPTION: 0.8,
        PositionKind.BODY: 0.0,
        PositionKind.OTHER: 0.0,
    }
    position_score = max(
        (position_values[position] for position in evidence.positions),
        default=0.0,
    )
    weights = policy.weights
    weighted = (
        weights.frequency * frequency_score
        + weights.document_dispersion * dispersion_score
        + weights.extraction_confidence * evidence.extraction_confidence
        + weights.resolution_confidence * evidence.resolution_confidence
        + weights.relation_participation * relation_score
        + weights.salient_position * position_score
    )
    total_weight = sum(getattr(weights, name) for name in weights.__dataclass_fields__)
    return min(1.0, max(0.0, weighted / total_weight))


def _ontology_type(ontology: object, entity_type: str):
    entity_types = getattr(ontology, "entity_types", None)
    if not isinstance(entity_types, Mapping):
        raise ValueError("ontology must expose an entity_types mapping")
    direct = entity_types.get(entity_type)
    if direct is not None:
        return direct
    for definition in entity_types.values():
        aliases = getattr(definition, "aliases", ())
        if entity_type in aliases:
            return definition
    return None


def decide_entity_filter(
    evidence: EntityFilterInput,
    ontology: object,
    policy: FilterPolicy,
) -> EntityFilterDecision:
    """Return a status-only policy decision while retaining raw evidence IDs."""

    if type(evidence) is not EntityFilterInput:
        raise ValueError("evidence must be an exact EntityFilterInput value")
    evidence.__post_init__()
    if type(policy) is not FilterPolicy:
        raise ValueError("policy must be an exact FilterPolicy value")
    policy.__post_init__()
    ontology_checksum = getattr(ontology, "checksum", None)
    if type(ontology_checksum) is not str or len(ontology_checksum) != 64:
        raise ValueError("ontology must expose a SHA-256 checksum")
    definition = _ontology_type(ontology, evidence.entity_type)
    base_utility = score_retrieval_utility(evidence, policy)
    ontology_weight = (
        1.0
        if definition is None
        else _unit_float(
            getattr(definition, "default_retrieval_weight", 1.0),
            "ontology retrieval weight",
        )
    )
    utility = base_utility * ontology_weight

    provenance = getattr(ontology, "provenance", {})
    provenance_extension_types = frozenset(
        value
        for value in str(
            provenance.get("enabled_entity_types", "")
            if isinstance(provenance, Mapping)
            else ""
        ).split(",")
        if value
    )
    enabled_extension_types = provenance_extension_types
    publisher_enabled = (
        evidence.entity_type == "publisher"
        and definition is not None
        and "publisher" in enabled_extension_types
    )
    if evidence.entity_type in policy.rejected_entity_types:
        status = FilterStatus.REJECTED
        reasons = ("entity_type_rejected_by_policy",)
    elif evidence.entity_type == "publisher" and not publisher_enabled:
        status = FilterStatus.SUPPRESSED
        reasons = ("publisher_suppressed_by_default",)
    elif definition is None:
        status = FilterStatus.REJECTED
        reasons = ("entity_type_not_in_ontology",)
    elif (
        evidence.entity_type in policy.suppressed_entity_types and not publisher_enabled
    ):
        status = FilterStatus.SUPPRESSED
        reasons = ("entity_type_suppressed_by_policy",)
    else:
        ontology_policy = str(
            getattr(definition, "default_suppression_policy", "below_confidence")
        )
        ontology_threshold = _unit_float(
            getattr(definition, "default_suppression_threshold", 0.0),
            "ontology suppression threshold",
        )
        if ontology_policy in {"reject", "rejected"}:
            status = FilterStatus.REJECTED
            reasons = ("ontology_rejection_policy",)
        elif ontology_policy in {"suppress", "suppressed"}:
            status = FilterStatus.SUPPRESSED
            reasons = ("ontology_suppression_policy",)
        elif utility < ontology_threshold:
            status = FilterStatus.SUPPRESSED
            reasons = ("below_ontology_utility_threshold",)
        elif utility < policy.utility_activation_threshold:
            status = FilterStatus.SUPPRESSED
            reasons = ("below_policy_utility_threshold",)
        else:
            status = FilterStatus.ACTIVE
            reasons = ("meets_utility_threshold",)
    return EntityFilterDecision(
        entity_id=evidence.entity_id,
        status=status,
        reason_codes=reasons,
        policy_version=policy.version,
        policy_checksum=filter_policy_checksum(policy),
        ontology_checksum=ontology_checksum,
        retained_mention_ids=evidence.mention_ids,
        extraction_confidence=evidence.extraction_confidence,
        resolution_confidence=evidence.resolution_confidence,
        retrieval_utility=utility,
        promotion_confidence=evidence.promotion_confidence,
    )


def filter_collection_entities(
    entities: Iterable[EntityFilterInput],
    ontology: object,
    policy: FilterPolicy,
) -> tuple[EntityFilterDecision, ...]:
    """Apply a policy deterministically without invoking extraction or an LLM."""

    bounded = tuple(entities)
    if len(bounded) > 50_000:
        raise ValueError("filter input exceeds the configured entity limit")
    if any(type(entity) is not EntityFilterInput for entity in bounded):
        raise ValueError("filter input must contain exact EntityFilterInput values")
    ordered = tuple(sorted(bounded, key=lambda entity: entity.entity_id))
    entity_ids = tuple(entity.entity_id for entity in ordered)
    if len(entity_ids) != len(set(entity_ids)):
        raise ValueError("duplicate entity id in filter input")
    mention_ids = tuple(
        mention_id for entity in ordered for mention_id in entity.mention_ids
    )
    if len(mention_ids) != len(set(mention_ids)):
        raise ValueError("duplicate mention id across filter entities")
    return tuple(decide_entity_filter(entity, ontology, policy) for entity in ordered)


def _explicit_position(metadata: object) -> PositionKind:
    """Use only persisted section evidence; never infer title/abstract from offsets."""

    if type(metadata) is not dict:
        return PositionKind.OTHER
    raw = metadata.get("position_kind")
    if type(raw) is not str:
        return PositionKind.OTHER
    try:
        return PositionKind(raw)
    except ValueError:
        return PositionKind.OTHER


def _filter_inputs_from_artifact(artifact):
    from django.db.models import Q

    from apps.knowledge_graph.models import (
        CollectionEntity,
        CollectionEntityDocumentLink,
        DocumentEntityMention,
        RelationMention,
    )

    entities = tuple(
        CollectionEntity.objects.select_for_update()
        .filter(artifact=artifact)
        .order_by("cluster_key")
    )
    automatic_links = tuple(
        CollectionEntityDocumentLink.objects.select_for_update()
        .select_related("document_entity")
        .filter(
            artifact=artifact,
            outcome=CollectionEntityDocumentLink.Outcome.AUTOMATIC,
        )
        .order_by("collection_entity_id", "document_entity_id")
    )
    links_by_entity: dict[int, list[object]] = {}
    document_entity_ids: list[int] = []
    for link in automatic_links:
        links_by_entity.setdefault(link.collection_entity_id, []).append(link)
        document_entity_ids.append(link.document_entity_id)
    memberships = tuple(
        DocumentEntityMention.objects.select_for_update()
        .select_related("mention")
        .filter(
            document_entity_id__in=document_entity_ids,
            status=DocumentEntityMention.Status.ACTIVE,
        )
        .order_by("document_entity_id", "mention_id")
    )
    memberships_by_document_entity: dict[int, list[object]] = {}
    all_mention_ids: set[int] = set()
    for membership in memberships:
        memberships_by_document_entity.setdefault(
            membership.document_entity_id, []
        ).append(membership)
        all_mention_ids.add(membership.mention_id)
    participation: dict[int, int] = {mention_id: 0 for mention_id in all_mention_ids}
    for head_id, tail_id in (
        RelationMention.objects.select_for_update()
        .filter(Q(head_id__in=all_mention_ids) | Q(tail_id__in=all_mention_ids))
        .values_list("head_id", "tail_id")
    ):
        if head_id in participation:
            participation[head_id] += 1
        if tail_id in participation:
            participation[tail_id] += 1
    projected: list[EntityFilterInput] = []
    for entity in entities:
        entity_memberships = tuple(
            membership
            for link in links_by_entity.get(entity.pk, ())
            for membership in memberships_by_document_entity.get(
                link.document_entity_id, ()
            )
        )
        if not entity_memberships:
            raise ValueError("collection entity has no retained raw mention evidence")
        projected.append(
            EntityFilterInput(
                entity_id=str(entity.pk),
                entity_type=entity.entity_type,
                mention_ids=tuple(str(item.mention_id) for item in entity_memberships),
                document_ids=tuple(
                    str(item.mention.document_id) for item in entity_memberships
                ),
                extraction_confidence=entity.extraction_confidence,
                resolution_confidence=entity.resolution_confidence,
                promotion_confidence=entity.promotion_confidence,
                relation_participation=sum(
                    participation[item.mention_id] for item in entity_memberships
                ),
                positions=tuple(
                    _explicit_position(item.mention.metadata)
                    for item in entity_memberships
                ),
            )
        )
    return entities, tuple(projected)


def create_filter_rerun_artifact(
    source_artifact_id: int,
    policy: FilterPolicy,
    ontology: object,
):
    """Clone resolution evidence into a new filtered shadow artifact.

    This operation intentionally never updates the active source artifact and
    never invokes GLiNER2 or an embedding provider.
    """

    from django.db import transaction

    from apps.knowledge_graph.models import (
        CollectionArtifactInput,
        CollectionEntity,
        CollectionEntityDocumentLink,
        GraphArtifact,
        GraphBuildRun,
    )
    from apps.knowledge_graph.resolution.collection import (
        _collection_entity_row_audit,
        _collection_link_row_audit,
        _snapshot_from_locked_manifest,
    )

    if type(source_artifact_id) is not int or source_artifact_id <= 0:
        raise ValueError("source artifact id must be a positive integer")
    if type(policy) is not FilterPolicy:
        raise ValueError("policy must be an exact FilterPolicy value")
    checksum = filter_policy_checksum(policy)
    ontology_checksum = getattr(ontology, "checksum", None)
    if type(ontology_checksum) is not str or len(ontology_checksum) != 64:
        raise ValueError("ontology must expose a SHA-256 checksum")
    with transaction.atomic():
        source = GraphArtifact.objects.select_for_update().get(pk=source_artifact_id)
        if (
            source.scope_type != GraphArtifact.ScopeType.COLLECTION
            or source.status != GraphArtifact.Status.ACTIVE
        ):
            raise ValueError(
                "filter rerun source must be an active collection artifact"
            )
        source_metadata = source.metadata if type(source.metadata) is dict else {}
        if source_metadata.get("ontology_checksum") != ontology_checksum:
            raise ValueError("filter ontology does not match source artifact")
        source_manifest = tuple(
            CollectionArtifactInput.objects.select_for_update()
            .select_related("document_artifact", "collection")
            .filter(artifact=source)
            .order_by("document_artifact_id")
        )
        _snapshot_from_locked_manifest(source, source_manifest)
        source_entities, evidence = _filter_inputs_from_artifact(source)
        decisions = filter_collection_entities(evidence, ontology, policy)
        decisions_by_id = {int(item.entity_id): item for item in decisions}
        destination = GraphArtifact.objects.create(
            scope_type=source.scope_type,
            scope_id=source.scope_id,
            status=GraphArtifact.Status.BUILDING,
            source_hash=source.source_hash,
            ontology_version=source.ontology_version,
            extractor_version=source.extractor_version,
            resolver_version=source.resolver_version,
            filter_policy_version=policy.version,
            embedding_model_signature=source.embedding_model_signature,
            metadata={
                **source_metadata,
                "filter_policy_checksum": checksum,
                "filter_source_artifact_id": source.pk,
            },
        )
        manifest_rows = [
            CollectionArtifactInput(
                artifact=destination,
                collection=row.collection,
                document_id=row.document_id,
                document_artifact=row.document_artifact,
                source_signature=row.source_signature,
                build_signature="0" * 64,
            )
            for row in source_manifest
        ]
        CollectionArtifactInput.objects.bulk_create(manifest_rows)
        new_entities = []
        for row in source_entities:
            decision = decisions_by_id[row.pk]
            new_entities.append(
                CollectionEntity(
                    artifact=destination,
                    collection_id=row.collection_id,
                    cluster_key=row.cluster_key,
                    label=row.label,
                    normalized_label=row.normalized_label,
                    version_signature=row.version_signature,
                    entity_type=row.entity_type,
                    identifier=row.identifier,
                    status=decision.status.value,
                    extraction_confidence=decision.extraction_confidence,
                    resolution_confidence=decision.resolution_confidence,
                    retrieval_utility=decision.retrieval_utility,
                    promotion_confidence=decision.promotion_confidence or 0.0,
                    filter_reason=decision.reason_codes[0],
                    embedding_model_signature=row.embedding_model_signature,
                    embedding_input_hash=row.embedding_input_hash,
                    embedding=row.embedding,
                    metadata={
                        **{
                            key: value
                            for key, value in (
                                row.metadata.items()
                                if type(row.metadata) is dict
                                else ()
                            )
                            if key != "row_audit_checksum"
                        },
                        "filter_policy_checksum": checksum,
                        "filter_reason_codes": list(decision.reason_codes),
                        "filter_source_entity_id": row.pk,
                    },
                )
            )
        for row in new_entities:
            row.metadata = {
                **row.metadata,
                "row_audit_checksum": _collection_entity_row_audit(row),
            }
        CollectionEntity.objects.bulk_create(new_entities)
        new_entity_by_old = dict(
            zip((row.pk for row in source_entities), new_entities, strict=True)
        )
        manifest_by_source_artifact = {
            row.document_artifact_id: row for row in manifest_rows
        }
        source_links = tuple(
            CollectionEntityDocumentLink.objects.select_for_update()
            .select_related("document_entity", "manifest_input")
            .filter(artifact=source)
            .order_by("pk")
        )
        cloned_links = [
            CollectionEntityDocumentLink(
                artifact=destination,
                manifest_input=manifest_by_source_artifact[
                    row.document_entity.artifact_id
                ],
                document_entity=row.document_entity,
                collection_entity=new_entity_by_old[row.collection_entity_id],
                score=row.score,
                identifier_score=row.identifier_score,
                alias_score=row.alias_score,
                embedding_similarity=row.embedding_similarity,
                neighborhood_agreement=row.neighborhood_agreement,
                method=row.method,
                resolver_version=row.resolver_version,
                outcome=row.outcome,
                candidate_rank=row.candidate_rank,
                decision_checksum=sha256(
                    f"{row.decision_checksum}:{destination.pk}:{checksum}".encode()
                ).hexdigest(),
                status=row.status,
                reason=row.reason,
                metadata={
                    **{
                        key: value
                        for key, value in (
                            row.metadata.items() if type(row.metadata) is dict else ()
                        )
                        if key != "row_audit_checksum"
                    },
                    "filter_source_link_id": row.pk,
                },
            )
            for row in source_links
        ]
        for row in cloned_links:
            row.metadata = {
                **row.metadata,
                "row_audit_checksum": _collection_link_row_audit(row),
            }
        CollectionEntityDocumentLink.objects.bulk_create(cloned_links)
        GraphBuildRun.objects.create(
            artifact=destination,
            stage=GraphBuildRun.Stage.FILTERING,
            status=GraphBuildRun.Status.SUCCEEDED,
            attempt=1,
            stats={
                "filter_commit": {
                    "version": 1,
                    "policy_checksum": checksum,
                    "source_artifact_id": source.pk,
                    "entity_count": len(new_entities),
                    "link_count": len(cloned_links),
                }
            },
        )
        return destination


__all__ = [
    "EntityFilterDecision",
    "EntityFilterInput",
    "FilterPolicy",
    "FilterStatus",
    "PositionKind",
    "UtilityWeights",
    "decide_entity_filter",
    "filter_collection_entities",
    "filter_policy_checksum",
    "create_filter_rerun_artifact",
    "score_retrieval_utility",
]
