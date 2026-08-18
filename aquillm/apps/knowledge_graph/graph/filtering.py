"""Transparent, status-only filtering for collection graph entities."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from math import isfinite, log1p

_HASH_PATTERN = re.compile(r"[0-9a-f]{64}")


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

    def __post_init__(self) -> None:
        object.__setattr__(self, "entity_id", _text(self.entity_id, "entity id"))
        if type(self.status) is not FilterStatus:
            raise ValueError("filter status must be exact FilterStatus")
        if type(self.reason_codes) is not tuple or not self.reason_codes:
            raise ValueError("filter reasons must be a nonempty exact tuple")
        reasons = tuple(
            sorted(_text(value, "filter reason", 128) for value in self.reason_codes)
        )
        if len(reasons) != len(set(reasons)):
            raise ValueError("filter reasons must be unique")
        object.__setattr__(self, "reason_codes", reasons)
        object.__setattr__(
            self, "policy_version", _text(self.policy_version, "policy version", 128)
        )
        for name in ("policy_checksum", "ontology_checksum"):
            value = getattr(self, name)
            if type(value) is not str or _HASH_PATTERN.fullmatch(value) is None:
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if (
            type(self.retained_mention_ids) is not tuple
            or not self.retained_mention_ids
        ):
            raise ValueError("retained mention IDs must be a nonempty exact tuple")
        retained = tuple(
            _text(value, "retained mention id") for value in self.retained_mention_ids
        )
        if len(retained) != len(set(retained)):
            raise ValueError("retained mention IDs must be unique")
        object.__setattr__(self, "retained_mention_ids", retained)
        for name in (
            "extraction_confidence",
            "resolution_confidence",
            "retrieval_utility",
        ):
            object.__setattr__(self, name, _unit_float(getattr(self, name), name))
        if self.promotion_confidence is not None:
            object.__setattr__(
                self,
                "promotion_confidence",
                _unit_float(self.promotion_confidence, "promotion confidence"),
            )


@dataclass(frozen=True, slots=True)
class CollectionFilterResult:
    """Checksummed status projection bound to one immutable resolution result."""

    resolution_checksum: str
    policy_version: str
    policy_checksum: str
    ontology_checksum: str
    inputs: tuple[EntityFilterInput, ...]
    decisions: tuple[EntityFilterDecision, ...]
    checksum: str

    def __post_init__(self) -> None:
        for name in (
            "resolution_checksum",
            "policy_checksum",
            "ontology_checksum",
        ):
            value = getattr(self, name)
            if type(value) is not str or _HASH_PATTERN.fullmatch(value) is None:
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        object.__setattr__(
            self, "policy_version", _text(self.policy_version, "policy version", 128)
        )
        if type(self.inputs) is not tuple or any(
            type(value) is not EntityFilterInput for value in self.inputs
        ):
            raise ValueError("filter inputs must be an exact typed tuple")
        if type(self.decisions) is not tuple or any(
            type(value) is not EntityFilterDecision for value in self.decisions
        ):
            raise ValueError("filter decisions must be an exact typed tuple")
        for value in self.inputs:
            value.__post_init__()
        for value in self.decisions:
            value.__post_init__()
        input_ids = tuple(value.entity_id for value in self.inputs)
        decision_ids = tuple(value.entity_id for value in self.decisions)
        if input_ids != tuple(sorted(set(input_ids))):
            raise ValueError("filter inputs must use unique stable entity order")
        if decision_ids != input_ids:
            raise ValueError("filter decisions must cover every input exactly once")
        for evidence, decision in zip(self.inputs, self.decisions, strict=True):
            if (
                decision.policy_version != self.policy_version
                or decision.policy_checksum != self.policy_checksum
                or decision.ontology_checksum != self.ontology_checksum
                or decision.retained_mention_ids != evidence.mention_ids
                or decision.extraction_confidence != evidence.extraction_confidence
                or decision.resolution_confidence != evidence.resolution_confidence
                or decision.promotion_confidence != evidence.promotion_confidence
            ):
                raise ValueError("filter decision audit does not match immutable input")
        if self.checksum:
            if _HASH_PATTERN.fullmatch(self.checksum) is None:
                raise ValueError("filter result checksum must be SHA-256")
            if collection_filter_result_checksum(self) != self.checksum:
                raise ValueError("filter result checksum is invalid")


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


def _filter_result_content(result: CollectionFilterResult) -> dict[str, object]:
    return {
        "resolution_checksum": result.resolution_checksum,
        "policy_version": result.policy_version,
        "policy_checksum": result.policy_checksum,
        "ontology_checksum": result.ontology_checksum,
        "inputs": [
            {
                "entity_id": item.entity_id,
                "entity_type": item.entity_type,
                "mention_ids": list(item.mention_ids),
                "document_ids": list(item.document_ids),
                "extraction_confidence": item.extraction_confidence,
                "resolution_confidence": item.resolution_confidence,
                "promotion_confidence": item.promotion_confidence,
                "relation_participation": item.relation_participation,
                "positions": [position.value for position in item.positions],
            }
            for item in result.inputs
        ],
        "decisions": [
            {
                "entity_id": item.entity_id,
                "status": item.status.value,
                "reason_codes": list(item.reason_codes),
                "policy_version": item.policy_version,
                "policy_checksum": item.policy_checksum,
                "ontology_checksum": item.ontology_checksum,
                "retained_mention_ids": list(item.retained_mention_ids),
                "extraction_confidence": item.extraction_confidence,
                "resolution_confidence": item.resolution_confidence,
                "retrieval_utility": item.retrieval_utility,
                "promotion_confidence": item.promotion_confidence,
            }
            for item in result.decisions
        ],
    }


def collection_filter_result_checksum(result: CollectionFilterResult) -> str:
    if type(result) is not CollectionFilterResult:
        raise ValueError("filter result must be exact CollectionFilterResult")
    return sha256(
        json.dumps(
            _filter_result_content(result),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def filter_collection_resolution(
    resolution: object,
    entities: Iterable[EntityFilterInput],
    ontology: object,
    policy: FilterPolicy,
) -> CollectionFilterResult:
    """Filter one resolution result without mutating identity or raw mentions."""

    from apps.knowledge_graph.resolution.collection import CollectionResolutionResult

    if type(resolution) is not CollectionResolutionResult:
        raise ValueError("resolution must be exact CollectionResolutionResult")
    resolution.__post_init__()
    if type(policy) is not FilterPolicy:
        raise ValueError("policy must be exact FilterPolicy")
    policy.__post_init__()
    policy_checksum = filter_policy_checksum(policy)
    if policy_checksum != resolution.snapshot.filter_policy_checksum:
        raise ValueError("filter policy checksum does not match resolution snapshot")
    from apps.knowledge_graph.services.ontology import validate_ontology_definition

    validate_ontology_definition(
        ontology,
        expected_version=resolution.snapshot.ontology_version,
        expected_checksum=resolution.snapshot.ontology_checksum,
    )
    ontology_checksum = ontology.checksum
    ordered = tuple(sorted(tuple(entities), key=lambda item: item.entity_id))
    if any(type(item) is not EntityFilterInput for item in ordered):
        raise ValueError("filter entities must contain exact EntityFilterInput values")
    clusters = {cluster.cluster_key: cluster for cluster in resolution.clusters}
    if tuple(item.entity_id for item in ordered) != tuple(sorted(clusters)):
        raise ValueError("filter evidence must cover every resolution cluster")
    for evidence in ordered:
        evidence.__post_init__()
        cluster = clusters[evidence.entity_id]
        if (
            evidence.entity_type != cluster.entity_type
            or evidence.extraction_confidence != cluster.extraction_confidence
            or evidence.resolution_confidence != cluster.resolution_confidence
        ):
            raise ValueError("filter evidence confidence/type differs from resolution")
    decisions = filter_collection_entities(ordered, ontology, policy)
    provisional = CollectionFilterResult(
        resolution_checksum=resolution.checksum,
        policy_version=policy.version,
        policy_checksum=policy_checksum,
        ontology_checksum=ontology_checksum,
        inputs=ordered,
        decisions=decisions,
        checksum="",
    )
    return CollectionFilterResult(
        resolution_checksum=provisional.resolution_checksum,
        policy_version=provisional.policy_version,
        policy_checksum=provisional.policy_checksum,
        ontology_checksum=provisional.ontology_checksum,
        inputs=provisional.inputs,
        decisions=provisional.decisions,
        checksum=collection_filter_result_checksum(provisional),
    )


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
    """Validate ontology semantics and return one deterministic filter decision."""

    from apps.knowledge_graph.services.ontology import validate_ontology_definition

    validate_ontology_definition(ontology)
    return _decide_entity_filter_validated(evidence, ontology, policy)


def _decide_entity_filter_validated(
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

    publisher_enabled = (
        evidence.entity_type == "publisher"
        and definition is not None
        and getattr(definition, "extension_enabled", False) is True
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
    from apps.knowledge_graph.services.ontology import validate_ontology_definition

    validate_ontology_definition(ontology)
    ordered = tuple(sorted(bounded, key=lambda entity: entity.entity_id))
    entity_ids = tuple(entity.entity_id for entity in ordered)
    if len(entity_ids) != len(set(entity_ids)):
        raise ValueError("duplicate entity id in filter input")
    mention_ids = tuple(
        mention_id for entity in ordered for mention_id in entity.mention_ids
    )
    if len(mention_ids) != len(set(mention_ids)):
        raise ValueError("duplicate mention id across filter entities")
    return tuple(
        _decide_entity_filter_validated(entity, ontology, policy) for entity in ordered
    )


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


def _filter_rerun_projection_checksum(
    *,
    source_artifact,
    policy_checksum: str,
    ontology_checksum: str,
    evidence: tuple[EntityFilterInput, ...],
    decisions: tuple[EntityFilterDecision, ...],
) -> str:
    payload = {
        "source_artifact_id": source_artifact.pk,
        "source_hash": source_artifact.source_hash,
        "resolution_config_checksum": source_artifact.resolution_config_checksum,
        "policy_checksum": policy_checksum,
        "ontology_checksum": ontology_checksum,
        "evidence": [
            {
                "entity_id": item.entity_id,
                "entity_type": item.entity_type,
                "mention_ids": list(item.mention_ids),
                "document_ids": list(item.document_ids),
                "extraction_confidence": item.extraction_confidence,
                "resolution_confidence": item.resolution_confidence,
                "promotion_confidence": item.promotion_confidence,
                "relation_participation": item.relation_participation,
                "positions": [position.value for position in item.positions],
            }
            for item in evidence
        ],
        "decisions": [
            {
                "entity_id": item.entity_id,
                "status": item.status.value,
                "reason_codes": list(item.reason_codes),
                "retrieval_utility": item.retrieval_utility,
                "promotion_confidence": item.promotion_confidence,
            }
            for item in decisions
        ],
    }
    return sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _filter_rerun_entity_matches(
    row,
    *,
    source,
    destination,
    decision: EntityFilterDecision,
    policy_checksum: str,
    projection_checksum: str,
) -> bool:
    """Compare every cloned entity field with its deterministic projection."""

    from apps.knowledge_graph.resolution.collection import (
        _collection_entity_row_audit,
    )
    from apps.knowledge_graph.resolution.scoring import validate_embedding

    def embedding(value):
        if value is None:
            return None
        try:
            return validate_embedding(tuple(float(component) for component in value))
        except (TypeError, ValueError):
            return object()

    source_metadata = source.metadata if type(source.metadata) is dict else {}
    expected_metadata = {
        **{
            key: value
            for key, value in source_metadata.items()
            if key != "row_audit_checksum"
        },
        "filter_policy_checksum": policy_checksum,
        "filter_result_checksum": projection_checksum,
        "filter_reason_codes": list(decision.reason_codes),
        "filter_source_entity_id": source.pk,
    }
    metadata = row.metadata if type(row.metadata) is dict else {}
    metadata_without_audit = {
        key: value for key, value in metadata.items() if key != "row_audit_checksum"
    }
    return bool(
        row.artifact_id == destination.pk
        and row.collection_id == source.collection_id
        and row.cluster_key == source.cluster_key
        and row.label == source.label
        and row.normalized_label == source.normalized_label
        and row.version_signature == source.version_signature
        and row.entity_type == source.entity_type
        and row.identifier == source.identifier
        and row.status == decision.status.value
        and row.extraction_confidence == decision.extraction_confidence
        and row.resolution_confidence == decision.resolution_confidence
        and row.retrieval_utility == decision.retrieval_utility
        and row.promotion_confidence == decision.promotion_confidence
        and row.filter_reason == decision.reason_codes[0]
        and row.embedding_model_signature == source.embedding_model_signature
        and row.embedding_input_hash == source.embedding_input_hash
        and embedding(row.embedding) == embedding(source.embedding)
        and metadata_without_audit == expected_metadata
        and metadata.get("row_audit_checksum") == _collection_entity_row_audit(row)
    )


def _filter_rerun_link_matches(
    row,
    *,
    source,
    destination,
    expected_manifest,
    expected_collection_entity,
    policy_checksum: str,
    projection_checksum: str,
) -> bool:
    """Compare every cloned link field with its immutable source resolution link."""

    from apps.knowledge_graph.resolution.collection import (
        _collection_link_row_audit,
    )

    source_metadata = source.metadata if type(source.metadata) is dict else {}
    expected_metadata = {
        **{
            key: value
            for key, value in source_metadata.items()
            if key != "row_audit_checksum"
        },
        "filter_source_link_id": source.pk,
        "filter_result_checksum": projection_checksum,
    }
    metadata = row.metadata if type(row.metadata) is dict else {}
    metadata_without_audit = {
        key: value for key, value in metadata.items() if key != "row_audit_checksum"
    }
    expected_decision_checksum = sha256(
        f"{source.decision_checksum}:{destination.pk}:{policy_checksum}".encode()
    ).hexdigest()
    return bool(
        row.artifact_id == destination.pk
        and row.manifest_input_id == expected_manifest.pk
        and row.document_entity_id == source.document_entity_id
        and row.collection_entity_id == expected_collection_entity.pk
        and row.score == source.score
        and row.identifier_score == source.identifier_score
        and row.alias_score == source.alias_score
        and row.embedding_similarity == source.embedding_similarity
        and row.neighborhood_agreement == source.neighborhood_agreement
        and row.method == source.method
        and row.resolver_version == source.resolver_version
        and row.outcome == source.outcome
        and row.candidate_rank == source.candidate_rank
        and row.decision_checksum == expected_decision_checksum
        and row.status == source.status
        and row.reason == source.reason
        and metadata_without_audit == expected_metadata
        and metadata.get("row_audit_checksum") == _collection_link_row_audit(row)
    )


def _canonical_marker_checksum(marker: object) -> str:
    return sha256(
        json.dumps(
            marker,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _lock_filter_source_commit(
    *,
    source,
    source_manifest,
    source_entities,
    source_links,
    source_runs,
    preferred_run_id: int | None = None,
):
    """Bind a fully authenticated source run, preserving an existing binding."""

    from apps.knowledge_graph.graph.assembly import (
        CollectionGraphAssemblyError,
        _validate_task9_lineage,
    )
    def validate(run):
        stats = run.stats if type(run.stats) is dict else {}
        resolution = stats.get("collection_resolution_commit")
        filter_commit = stats.get("filter_commit")
        if (resolution is None) == (filter_commit is None):
            return None
        marker = resolution if resolution is not None else filter_commit
        if type(marker) is not dict or any(
            getattr(run, field) != getattr(source, field)
            for field in (
                "scope_type",
                "scope_id",
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
            )
        ):
            return None
        try:
            _validate_task9_lineage(
                source,
                run,
                source_manifest,
                source_entities,
                source_links,
            )
        except CollectionGraphAssemblyError:
            return None
        assembly_marker = stats.get("collection_assembly_commit")
        if assembly_marker is not None:
            if type(assembly_marker) is not dict:
                return None
            expected_assembly_checksum = _canonical_marker_checksum(
                {
                    key: value
                    for key, value in assembly_marker.items()
                    if key != "marker_checksum"
                }
            )
            if (
                type(assembly_marker.get("marker_checksum")) is not str
                or assembly_marker["marker_checksum"]
                != expected_assembly_checksum
            ):
                return None
        return (
            run,
            _canonical_marker_checksum(marker),
            None if assembly_marker is None else assembly_marker["marker_checksum"],
        )

    runs = tuple(source_runs)
    if preferred_run_id is not None:
        if type(preferred_run_id) is not int or preferred_run_id < 1:
            raise ValueError("existing filter source build run identity is invalid")
        preferred = next((row for row in runs if row.pk == preferred_run_id), None)
        result = None if preferred is None else validate(preferred)
        if result is None:
            raise ValueError("existing filter source build run is no longer valid")
        return result
    for run in sorted(runs, key=lambda row: (row.attempt, row.pk), reverse=True):
        result = validate(run)
        if result is not None:
            return result
    if not runs:
        raise ValueError("filter rerun source has no valid committed build run")
    raise ValueError("filter rerun source has no valid committed build run")


def _validate_existing_filter_rerun(
    *,
    destination,
    source,
    source_manifest,
    source_entities,
    source_links,
    decisions,
    policy_checksum,
    ontology_checksum,
    projection_checksum,
    source_build_run_id,
    source_task9_marker_checksum,
    source_assembly_marker_checksum,
    max_document_inputs,
):
    from apps.knowledge_graph.models import (
        CollectionArtifactInput,
        CollectionEntity,
        CollectionEntityDocumentLink,
        GraphArtifact,
        GraphBuildRun,
    )
    from apps.knowledge_graph.resolution.collection import (
        _snapshot_from_locked_manifest,
    )

    if destination.status != GraphArtifact.Status.BUILDING:
        raise ValueError("existing filter rerun is not a building artifact")
    if destination.metadata != {
        "filter_source_artifact_id": source.pk,
        "filter_source_build_run_id": source_build_run_id,
        "filter_source_task9_marker_checksum": source_task9_marker_checksum,
        "filter_source_assembly_marker_checksum": source_assembly_marker_checksum,
        "filter_result_checksum": projection_checksum,
    }:
        raise ValueError("existing filter rerun artifact audit is corrupt")
    manifest_query = (
        CollectionArtifactInput.objects.select_for_update()
        .select_related("document_artifact", "collection")
        .filter(artifact=destination)
        .order_by("document_artifact_id")
    )
    if manifest_query.count() > max_document_inputs:
        raise ValueError("existing filter rerun manifest exceeds its input cap")
    manifest = tuple(manifest_query)
    if len(manifest) != len(source_manifest):
        raise ValueError("existing filter rerun is partial or corrupt")
    _snapshot_from_locked_manifest(destination, manifest)
    source_manifest_identity = tuple(
        (
            row.collection_id,
            row.document_id,
            row.document_artifact_id,
            row.membership_signature,
            row.source_signature,
        )
        for row in source_manifest
    )
    destination_manifest_identity = tuple(
        (
            row.collection_id,
            row.document_id,
            row.document_artifact_id,
            row.membership_signature,
            row.source_signature,
        )
        for row in manifest
    )
    if destination_manifest_identity != source_manifest_identity:
        raise ValueError("existing filter rerun manifest differs from source")
    runs = tuple(
        GraphBuildRun.objects.select_for_update()
        .filter(artifact=destination)
        .order_by("pk")
    )
    if len(runs) != 1:
        raise ValueError("existing filter rerun marker is missing or ambiguous")
    run = runs[0]
    if (
        run.artifact_id != destination.pk
        or run.stage != GraphBuildRun.Stage.FILTERING
        or run.status != GraphBuildRun.Status.SUCCEEDED
        or any(
            getattr(run, field) != getattr(destination, field)
            for field in (
                "scope_type",
                "scope_id",
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
            )
        )
    ):
        raise ValueError("existing filter rerun build snapshot is corrupt")
    marker = run.stats.get("filter_commit") if type(run.stats) is dict else None
    expected_marker = {
        "version": 1,
        "policy_checksum": policy_checksum,
        "ontology_checksum": ontology_checksum,
        "resolution_config_checksum": source.resolution_config_checksum,
        "max_document_inputs": max_document_inputs,
        "filter_result_checksum": projection_checksum,
        "source_artifact_id": source.pk,
        "source_build_run_id": source_build_run_id,
        "source_task9_marker_checksum": source_task9_marker_checksum,
        "source_assembly_marker_checksum": source_assembly_marker_checksum,
        "source_hash": source.source_hash,
        "assembly_version": destination.assembly_version,
        "assembly_config_checksum": destination.assembly_config_checksum,
        "manifest_count": len(source_manifest),
        "entity_count": len(source_entities),
        "link_count": len(source_links),
    }
    if marker != expected_marker:
        raise ValueError("existing filter rerun commit marker is corrupt")
    entities = tuple(
        CollectionEntity.objects.select_for_update()
        .filter(artifact=destination)
        .order_by("cluster_key")
    )
    links = tuple(
        CollectionEntityDocumentLink.objects.select_for_update()
        .select_related("collection_entity")
        .filter(artifact=destination)
        .order_by("pk")
    )
    if len(entities) != len(source_entities) or len(links) != len(source_links):
        raise ValueError("existing filter rerun rows are partial or corrupt")
    decisions_by_source = {int(item.entity_id): item for item in decisions}
    source_entities_by_id = {row.pk: row for row in source_entities}
    destination_entities_by_source_id = {}
    for row in entities:
        metadata = row.metadata if type(row.metadata) is dict else {}
        source_id = metadata.get("filter_source_entity_id")
        decision = decisions_by_source.get(source_id)
        source_entity = source_entities_by_id.get(source_id)
        if (
            decision is None
            or source_entity is None
            or not _filter_rerun_entity_matches(
                row,
                source=source_entity,
                destination=destination,
                decision=decision,
                policy_checksum=policy_checksum,
                projection_checksum=projection_checksum,
            )
        ):
            raise ValueError("existing filter rerun entity audit is corrupt")
        destination_entities_by_source_id[source_id] = row
    if set(destination_entities_by_source_id) != set(source_entities_by_id):
        raise ValueError("existing filter rerun entity source mapping is corrupt")
    expected_source_link_ids = {row.pk for row in source_links}
    actual_source_link_ids = set()
    source_links_by_id = {row.pk: row for row in source_links}
    manifest_by_source_artifact = {row.document_artifact_id: row for row in manifest}
    for row in links:
        metadata = row.metadata if type(row.metadata) is dict else {}
        source_link_id = metadata.get("filter_source_link_id")
        actual_source_link_ids.add(source_link_id)
        source_link = source_links_by_id.get(source_link_id)
        expected_manifest = (
            None
            if source_link is None
            else manifest_by_source_artifact.get(
                source_link.document_entity.artifact_id
            )
        )
        expected_collection_entity = (
            None
            if source_link is None
            else destination_entities_by_source_id.get(source_link.collection_entity_id)
        )
        if (
            source_link is None
            or expected_manifest is None
            or expected_collection_entity is None
            or not _filter_rerun_link_matches(
                row,
                source=source_link,
                destination=destination,
                expected_manifest=expected_manifest,
                expected_collection_entity=expected_collection_entity,
                policy_checksum=policy_checksum,
                projection_checksum=projection_checksum,
            )
        ):
            raise ValueError("existing filter rerun link audit is corrupt")
    if actual_source_link_ids != expected_source_link_ids:
        raise ValueError("existing filter rerun links do not match source")
    return destination


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

    from apps.knowledge_graph.graph.assembly import lock_collection_graph_scope
    from apps.knowledge_graph.models import (
        CollectionArtifactInput,
        CollectionEntity,
        CollectionEntityDocumentLink,
        GraphArtifact,
        GraphBuildRun,
    )
    from apps.knowledge_graph.resolution.collection import (
        MAX_COLLECTION_DOCUMENT_INPUTS,
        _collection_entity_row_audit,
        _collection_link_row_audit,
        _snapshot_from_locked_manifest,
    )

    if type(source_artifact_id) is not int or source_artifact_id <= 0:
        raise ValueError("source artifact id must be a positive integer")
    if type(policy) is not FilterPolicy:
        raise ValueError("policy must be an exact FilterPolicy value")
    checksum = filter_policy_checksum(policy)
    from apps.knowledge_graph.services.ontology import validate_ontology_definition

    validate_ontology_definition(ontology)
    ontology_checksum = ontology.checksum
    source_reference = GraphArtifact.objects.only("scope_type", "scope_id").get(
        pk=source_artifact_id
    )
    if source_reference.scope_type != GraphArtifact.ScopeType.COLLECTION:
        raise ValueError("filter rerun source must be a collection artifact")
    try:
        collection_id = int(source_reference.scope_id)
    except (TypeError, ValueError) as exc:
        raise ValueError("filter rerun source collection identity is invalid") from exc
    with transaction.atomic():
        lock_collection_graph_scope(collection_id)
        scope_artifacts = tuple(
            GraphArtifact.objects.select_for_update()
            .filter(
                scope_type=GraphArtifact.ScopeType.COLLECTION,
                scope_id=str(collection_id),
            )
            .order_by("pk")
        )
        source = next(
            (row for row in scope_artifacts if row.pk == source_artifact_id), None
        )
        if source is None or source.status != GraphArtifact.Status.ACTIVE:
            raise ValueError(
                "filter rerun source must be an active collection artifact"
            )
        scope_runs = tuple(
            GraphBuildRun.objects.select_for_update()
            .filter(
                artifact__scope_type=GraphArtifact.ScopeType.COLLECTION,
                artifact__scope_id=str(collection_id),
            )
            .order_by("pk")
        )
        if source.ontology_checksum != ontology_checksum:
            raise ValueError("filter ontology does not match source artifact")
        validate_ontology_definition(
            ontology,
            expected_version=source.ontology_version,
            expected_checksum=source.ontology_checksum,
        )
        source_manifest_query = (
            CollectionArtifactInput.objects.select_for_update()
            .select_related("document_artifact", "collection")
            .filter(artifact=source)
            .order_by("document_artifact_id")
        )
        if source_manifest_query.count() > MAX_COLLECTION_DOCUMENT_INPUTS:
            raise ValueError("filter source manifest exceeds the v1 input cap")
        source_manifest = tuple(source_manifest_query)
        _snapshot_from_locked_manifest(source, source_manifest)
        source_entities, evidence = _filter_inputs_from_artifact(source)
        decisions = filter_collection_entities(evidence, ontology, policy)
        decisions_by_id = {int(item.entity_id): item for item in decisions}
        source_links = tuple(
            CollectionEntityDocumentLink.objects.select_for_update()
            .select_related("document_entity", "manifest_input")
            .filter(artifact=source)
            .order_by("pk")
        )
        projection_checksum = _filter_rerun_projection_checksum(
            source_artifact=source,
            policy_checksum=checksum,
            ontology_checksum=ontology_checksum,
            evidence=evidence,
            decisions=decisions,
        )
        identity = {
            "scope_type": source.scope_type,
            "scope_id": source.scope_id,
            "source_hash": source.source_hash,
            "ontology_version": source.ontology_version,
            "extractor_version": source.extractor_version,
            "resolver_version": source.resolver_version,
            "filter_policy_version": policy.version,
            "embedding_model_signature": source.embedding_model_signature,
            "ontology_checksum": source.ontology_checksum,
            "filter_policy_checksum": checksum,
            "resolution_config_checksum": source.resolution_config_checksum,
            "assembly_version": source.assembly_version,
            "assembly_config_checksum": source.assembly_config_checksum,
        }
        existing = next(
            (
                row
                for row in scope_artifacts
                if all(getattr(row, key) == value for key, value in identity.items())
            ),
            None,
        )
        preferred_run_id = None
        if existing is not None:
            existing_metadata = (
                existing.metadata if type(existing.metadata) is dict else {}
            )
            preferred_run_id = existing_metadata.get("filter_source_build_run_id")
        (
            source_build_run,
            source_task9_marker_checksum,
            source_assembly_marker_checksum,
        ) = _lock_filter_source_commit(
            source=source,
            source_manifest=source_manifest,
            source_entities=source_entities,
            source_links=source_links,
            source_runs=tuple(
                run for run in scope_runs if run.artifact_id == source.pk
            ),
            preferred_run_id=preferred_run_id,
        )
        source_stats = (
            source_build_run.stats
            if type(source_build_run.stats) is dict
            else {}
        )
        source_marker = source_stats.get("collection_resolution_commit")
        if source_marker is None:
            source_marker = source_stats.get("filter_commit")
        manifest_cap = (
            None
            if type(source_marker) is not dict
            else source_marker.get("max_document_inputs")
        )
        if (
            type(manifest_cap) is not int
            or not 1 <= manifest_cap <= MAX_COLLECTION_DOCUMENT_INPUTS
            or len(source_manifest) > manifest_cap
        ):
            raise ValueError("filter source manifest cap is invalid")
        if existing is not None:
            return _validate_existing_filter_rerun(
                destination=existing,
                source=source,
                source_manifest=source_manifest,
                source_entities=source_entities,
                source_links=source_links,
                decisions=decisions,
                policy_checksum=checksum,
                ontology_checksum=ontology_checksum,
                projection_checksum=projection_checksum,
                source_build_run_id=source_build_run.pk,
                source_task9_marker_checksum=source_task9_marker_checksum,
                source_assembly_marker_checksum=source_assembly_marker_checksum,
                max_document_inputs=manifest_cap,
            )
        destination = GraphArtifact.objects.create(
            status=GraphArtifact.Status.BUILDING,
            metadata={
                "filter_source_artifact_id": source.pk,
                "filter_source_build_run_id": source_build_run.pk,
                "filter_source_task9_marker_checksum": (
                    source_task9_marker_checksum
                ),
                "filter_source_assembly_marker_checksum": (
                    source_assembly_marker_checksum
                ),
                "filter_result_checksum": projection_checksum,
            },
            **identity,
        )
        manifest_rows = [
            CollectionArtifactInput(
                artifact=destination,
                collection=row.collection,
                document_id=row.document_id,
                document_artifact=row.document_artifact,
                membership_signature=row.membership_signature,
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
                    promotion_confidence=decision.promotion_confidence,
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
                        "filter_result_checksum": projection_checksum,
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
                    "filter_result_checksum": projection_checksum,
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
                    "ontology_checksum": ontology_checksum,
                    "resolution_config_checksum": source.resolution_config_checksum,
                    "max_document_inputs": manifest_cap,
                    "filter_result_checksum": projection_checksum,
                    "source_artifact_id": source.pk,
                    "source_build_run_id": source_build_run.pk,
                    "source_task9_marker_checksum": (
                        source_task9_marker_checksum
                    ),
                    "source_assembly_marker_checksum": (
                        source_assembly_marker_checksum
                    ),
                    "source_hash": source.source_hash,
                    "assembly_version": destination.assembly_version,
                    "assembly_config_checksum": (
                        destination.assembly_config_checksum
                    ),
                    "manifest_count": len(source_manifest),
                    "entity_count": len(new_entities),
                    "link_count": len(cloned_links),
                }
            },
        )
        return destination


__all__ = [
    "CollectionFilterResult",
    "EntityFilterDecision",
    "EntityFilterInput",
    "FilterPolicy",
    "FilterStatus",
    "PositionKind",
    "UtilityWeights",
    "decide_entity_filter",
    "filter_collection_entities",
    "filter_collection_resolution",
    "filter_policy_checksum",
    "collection_filter_result_checksum",
    "create_filter_rerun_artifact",
    "score_retrieval_utility",
]
