"""Deterministic collection resolution over an immutable manifest snapshot.

Provider SDKs and the application's embedding implementation are imported only
inside the strict default embedding backend. Pure resolution therefore remains
safe to import in web workers that do not install the optional KG runtime.
"""

from __future__ import annotations

import json
import re
import unicodedata
import uuid
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite

from .normalization import normalize_entity_label
from .scoring import (
    EMBEDDING_DIMENSIONS,
    ResolutionOutcome,
    ResolutionThresholds,
    ResolutionTier,
    classify_resolution_score,
    combine_resolution_scores,
    cosine_similarity,
    validate_embedding,
)

COLLECTION_RESOLVER_VERSION = "collection-resolution-v1"
EMBEDDING_PREPROCESSING_VERSION = "kg-entity-v1"
MAX_COLLECTION_ENTITIES = 50_000
MAX_RELATIONS = 250_000
MAX_TEXT_CHARACTERS = 8_192
_HASH_PATTERN = re.compile(r"[0-9a-f]{64}")
_VERSION_PATTERN = re.compile(r"[a-z0-9][a-z0-9.+:/_-]*")
_ACRONYM_PATTERN = re.compile(r"[A-Z][A-Z0-9-]{1,11}")
_ALIAS_METHODS = frozenset(
    {"stable_identifier", "defined_acronym", "ontology_alias", "normalized_name"}
)


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or not 1 <= value <= 2**63 - 1:
        raise ValueError(f"{label} must be a positive database integer")
    return value


def _bounded_text(
    value: object,
    label: str,
    *,
    maximum: int = MAX_TEXT_CHARACTERS,
    allow_empty: bool = False,
) -> str:
    if type(value) is not str:
        raise ValueError(f"{label} must be an exact string")
    normalized = " ".join(unicodedata.normalize("NFC", value).split())
    if not normalized and not allow_empty:
        raise ValueError(f"{label} must be nonempty")
    if len(normalized) > maximum or "\x00" in normalized:
        raise ValueError(f"{label} is unsafe or exceeds {maximum} characters")
    return normalized


def _hash(value: object) -> str:
    return sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _require_hash(value: object, label: str) -> str:
    if type(value) is not str or not _HASH_PATTERN.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _unit_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number in [0, 1]")
    number = float(value)
    if not isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{label} must be a finite number in [0, 1]")
    return number


def _quantize(value: float | None) -> float | None:
    return None if value is None else round(value, 12)


def embedding_text_hash(text: object) -> str:
    """Hash the exact normalized input covered by the preprocessing signature."""

    normalized = _bounded_text(text, "embedding text")
    return sha256(normalized.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class SignedEmbeddingBatch:
    """Backend response binding vector order to the actual provider/model."""

    vectors: tuple[tuple[float, ...], ...]
    text_hashes: tuple[str, ...]
    model_signature: str


@dataclass(frozen=True, slots=True)
class EmbeddedText:
    text: str
    input_hash: str
    vector: tuple[float, ...]


EmbeddingBackend = Callable[[tuple[str, ...]], SignedEmbeddingBatch]


class CollectionEmbeddingSession:
    """One build-scoped, signature-locked embedding session with stable caching."""

    __slots__ = ("expected_model_signature", "_backend", "_cache")

    def __init__(
        self,
        *,
        expected_model_signature: str,
        backend: EmbeddingBackend,
    ) -> None:
        signature = _bounded_text(
            expected_model_signature, "embedding model signature", maximum=512
        )
        if "dims=1024" not in signature or "prep=" not in signature:
            raise ValueError(
                "embedding model signature must lock 1024 dimensions and preprocessing"
            )
        if not callable(backend):
            raise ValueError("embedding backend must be callable")
        self.expected_model_signature = signature
        self._backend = backend
        self._cache: dict[str, tuple[str, tuple[float, ...]]] = {}

    def embed(self, texts: Sequence[str]) -> tuple[EmbeddedText, ...]:
        if not isinstance(texts, (tuple, list)):
            raise ValueError("embedding inputs must be a concrete sequence")
        normalized = tuple(_bounded_text(text, "embedding text") for text in texts)
        missing = tuple(sorted(set(normalized).difference(self._cache)))
        if missing:
            batch = self._backend(missing)
            if type(batch) is not SignedEmbeddingBatch:
                raise ValueError("embedding backend must return SignedEmbeddingBatch")
            if batch.model_signature != self.expected_model_signature:
                raise ValueError("embedding provider/model signature drift detected")
            if len(batch.vectors) != len(missing) or len(batch.text_hashes) != len(
                missing
            ):
                raise ValueError("embedding backend must return one vector per input")
            expected_hashes = tuple(embedding_text_hash(text) for text in missing)
            if batch.text_hashes != expected_hashes:
                raise ValueError(
                    "embedding backend output order/hash does not match input"
                )
            for text, input_hash, vector in zip(
                missing, batch.text_hashes, batch.vectors, strict=True
            ):
                try:
                    validated = validate_embedding(vector)
                except ValueError as exc:
                    raise ValueError(
                        f"embedding vector must contain 1024 finite values: {exc}"
                    ) from exc
                self._cache[text] = (input_hash, validated)
        return tuple(
            EmbeddedText(
                text=text,
                input_hash=self._cache[text][0],
                vector=self._cache[text][1],
            )
            for text in normalized
        )


def default_collection_embedding_session(
    expected_model_signature: str,
) -> CollectionEmbeddingSession:
    """Create a strict local-only session without importing providers eagerly."""

    def backend(texts: tuple[str, ...]) -> SignedEmbeddingBatch:
        from aquillm.utils import get_strict_index_embeddings

        vectors, signature = get_strict_index_embeddings(
            list(texts), expected_model_signature=expected_model_signature
        )
        return SignedEmbeddingBatch(
            vectors=tuple(tuple(vector) for vector in vectors),
            text_hashes=tuple(embedding_text_hash(text) for text in texts),
            model_signature=signature,
        )

    return CollectionEmbeddingSession(
        expected_model_signature=expected_model_signature,
        backend=backend,
    )


@dataclass(frozen=True, slots=True)
class CollectionSnapshotInput:
    manifest_input_id: int
    document_artifact_id: int
    document_id: uuid.UUID
    source_signature: str
    build_signature: str

    def __post_init__(self) -> None:
        _positive_int(self.manifest_input_id, "manifest input id")
        _positive_int(self.document_artifact_id, "document artifact id")
        if type(self.document_id) is not uuid.UUID:
            raise ValueError("snapshot document id must be an exact UUID")
        _require_hash(self.source_signature, "source signature")
        _require_hash(self.build_signature, "build signature")


@dataclass(frozen=True, slots=True)
class CollectionBuildSnapshot:
    destination_artifact_id: int
    collection_id: int
    inputs: tuple[CollectionSnapshotInput, ...]
    source_hash: str
    ontology_checksum: str

    def __post_init__(self) -> None:
        _positive_int(self.destination_artifact_id, "destination artifact id")
        _positive_int(self.collection_id, "collection id")
        if type(self.inputs) is not tuple or any(
            type(item) is not CollectionSnapshotInput for item in self.inputs
        ):
            raise ValueError("snapshot inputs must be an exact typed tuple")
        for item in self.inputs:
            item.__post_init__()
        ordered = tuple(
            sorted(
                self.inputs,
                key=lambda item: (
                    item.document_artifact_id,
                    str(item.document_id),
                    item.manifest_input_id,
                ),
            )
        )
        if len({item.manifest_input_id for item in ordered}) != len(ordered):
            raise ValueError("snapshot manifest IDs must be unique")
        if len({item.document_artifact_id for item in ordered}) != len(ordered):
            raise ValueError("snapshot document artifacts must be unique")
        if len({item.document_id for item in ordered}) != len(ordered):
            raise ValueError("snapshot document IDs must be unique")
        object.__setattr__(self, "inputs", ordered)
        _require_hash(self.source_hash, "snapshot source hash")
        _require_hash(self.ontology_checksum, "ontology checksum")

    @property
    def document_artifact_ids(self) -> tuple[int, ...]:
        return tuple(item.document_artifact_id for item in self.inputs)


@dataclass(frozen=True, slots=True)
class AliasEvidence:
    alias: str
    method: str
    mention_id: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "alias", _bounded_text(self.alias, "alias", maximum=512)
        )
        method = _bounded_text(self.method, "alias method", maximum=64)
        if method not in _ALIAS_METHODS:
            raise ValueError("alias evidence method is not a Task 8 membership method")
        object.__setattr__(self, "method", method)
        _positive_int(self.mention_id, "alias mention id")


@dataclass(frozen=True, slots=True)
class DocumentEntityInput:
    entity_id: int
    document_cluster_key: str
    document_artifact_id: int
    document_id: uuid.UUID
    label: str
    normalized_label: str
    entity_type: str
    identifier: str = ""
    version_signature: str = ""
    alias_evidence: tuple[AliasEvidence, ...] = ()
    description: str = ""
    extraction_confidence: float = 1.0
    document_resolution_confidence: float = 1.0

    def __post_init__(self) -> None:
        _positive_int(self.entity_id, "document entity id")
        _require_hash(self.document_cluster_key, "document cluster key")
        _positive_int(self.document_artifact_id, "document artifact id")
        if type(self.document_id) is not uuid.UUID:
            raise ValueError("document id must be an exact UUID")
        object.__setattr__(self, "label", _bounded_text(self.label, "entity label"))
        object.__setattr__(
            self,
            "normalized_label",
            _bounded_text(self.normalized_label, "normalized label", maximum=512),
        )
        object.__setattr__(
            self,
            "entity_type",
            _bounded_text(self.entity_type, "entity type", maximum=128),
        )
        object.__setattr__(
            self,
            "identifier",
            _bounded_text(
                self.identifier, "stable identifier", maximum=255, allow_empty=True
            ),
        )
        version = _bounded_text(
            self.version_signature,
            "version signature",
            maximum=128,
            allow_empty=True,
        )
        if version and not _VERSION_PATTERN.fullmatch(version):
            raise ValueError("version signature must use canonical lower-ASCII tokens")
        object.__setattr__(self, "version_signature", version)
        if type(self.alias_evidence) is not tuple or any(
            type(alias) is not AliasEvidence for alias in self.alias_evidence
        ):
            raise ValueError("alias evidence must be an exact typed tuple")
        for alias in self.alias_evidence:
            alias.__post_init__()
        aliases = tuple(
            sorted(
                self.alias_evidence,
                key=lambda item: (item.alias.casefold(), item.alias, item.mention_id),
            )
        )
        if len({item.mention_id for item in aliases}) != len(aliases):
            raise ValueError("alias evidence mention IDs must be unique")
        object.__setattr__(self, "alias_evidence", aliases)
        object.__setattr__(
            self,
            "description",
            _bounded_text(
                self.description,
                "entity description",
                allow_empty=True,
            ),
        )
        object.__setattr__(
            self,
            "extraction_confidence",
            _unit_float(self.extraction_confidence, "extraction confidence"),
        )
        object.__setattr__(
            self,
            "document_resolution_confidence",
            _unit_float(
                self.document_resolution_confidence,
                "document resolution confidence",
            ),
        )


@dataclass(frozen=True, slots=True)
class SupportedRelation:
    relation_id: int
    source_entity_id: int
    relation_type: str
    target_entity_id: int
    confidence: float
    supported: bool = True

    def __post_init__(self) -> None:
        _positive_int(self.relation_id, "relation id")
        _positive_int(self.source_entity_id, "relation source id")
        _positive_int(self.target_entity_id, "relation target id")
        if self.source_entity_id == self.target_entity_id:
            raise ValueError("relation endpoints must be distinct")
        object.__setattr__(
            self,
            "relation_type",
            _bounded_text(self.relation_type, "relation type", maximum=128),
        )
        object.__setattr__(
            self, "confidence", _unit_float(self.confidence, "relation confidence")
        )
        if type(self.supported) is not bool:
            raise ValueError("supported must be an exact bool")


@dataclass(frozen=True, slots=True)
class CollectionResolutionConfig:
    thresholds: ResolutionThresholds = ResolutionThresholds()
    max_candidates_per_entity: int = 16
    max_candidate_pool_per_entity: int = 128
    exact_semantic_scan_limit: int = 512
    embedding_weight: float = 0.85
    neighborhood_weight: float = 0.15
    relation_support_threshold: float = 0.50
    max_entities: int = MAX_COLLECTION_ENTITIES

    def __post_init__(self) -> None:
        if type(self.thresholds) is not ResolutionThresholds:
            raise ValueError("thresholds must be exact ResolutionThresholds")
        self.thresholds.__post_init__()
        for name, lower, upper in (
            ("max_candidates_per_entity", 1, 1_000),
            ("max_candidate_pool_per_entity", 1, 10_000),
            ("exact_semantic_scan_limit", 2, 10_000),
            ("max_entities", 1, MAX_COLLECTION_ENTITIES),
        ):
            value = getattr(self, name)
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
        if self.max_candidate_pool_per_entity < self.max_candidates_per_entity:
            raise ValueError("candidate pool must be at least candidate fanout")
        weights = []
        for name in ("embedding_weight", "neighborhood_weight"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be finite and nonnegative")
            number = float(value)
            if not isfinite(number) or number < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, number)
            weights.append(number)
        if not sum(weights):
            raise ValueError("resolution score weights must have positive sum")
        object.__setattr__(
            self,
            "relation_support_threshold",
            _unit_float(self.relation_support_threshold, "relation support threshold"),
        )


@dataclass(frozen=True, slots=True)
class CollectionPairDecision:
    left_entity_id: int
    right_entity_id: int
    outcome: ResolutionOutcome
    tier: ResolutionTier
    resolution_confidence: float
    identifier_score: float | None
    alias_score: float | None
    embedding_similarity: float | None
    neighborhood_agreement: float | None
    reason_codes: tuple[str, ...]
    embedding_model_signature: str | None = None
    left_embedding_input_hash: str | None = None
    right_embedding_input_hash: str | None = None
    candidate_rank: int | None = None


@dataclass(frozen=True, slots=True)
class CollectionEntityCluster:
    cluster_key: str
    document_entity_ids: tuple[int, ...]
    document_cluster_keys: tuple[str, ...]
    document_artifact_ids: tuple[int, ...]
    document_ids: tuple[uuid.UUID, ...]
    label: str
    normalized_label: str
    version_signature: str
    entity_type: str
    identifier: str
    aliases: tuple[str, ...]
    embedding: tuple[float, ...] | None
    embedding_model_signature: str | None
    embedding_input_hash: str | None
    extraction_confidence: float
    resolution_confidence: float
    retrieval_utility: float | None
    promotion_confidence: float | None
    resolution_methods: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ResolutionAudit:
    embedding_model_signature: str
    max_candidates_per_entity: int
    max_candidate_pool_per_entity: int
    max_observed_candidate_fanout: int
    embedded_entity_count: int
    embedding_candidate_pair_count: int
    type_incompatible_pair_count: int
    automatic_pair_count: int
    candidate_pair_count: int
    rejected_pair_count: int


@dataclass(frozen=True, slots=True)
class CollectionResolutionResult:
    snapshot: CollectionBuildSnapshot
    resolver_version: str
    input_fingerprint: str
    source_entity_fingerprint: str
    source_relation_fingerprint: str
    source_entity_ids: tuple[int, ...]
    clusters: tuple[CollectionEntityCluster, ...]
    decisions: tuple[CollectionPairDecision, ...]
    audit: ResolutionAudit
    checksum: str


class _DisjointSet:
    def __init__(self, values: Iterable[int]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: int) -> int:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            next_value = self.parent[value]
            self.parent[value] = root
            value = next_value
        return root

    def union(self, left: int, right: int) -> int:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return left_root
        root, child = sorted((left_root, right_root))
        self.parent[child] = root
        return root

    def groups(self) -> dict[int, tuple[int, ...]]:
        result: dict[int, list[int]] = defaultdict(list)
        for value in sorted(self.parent):
            result[self.find(value)].append(value)
        return {root: tuple(values) for root, values in sorted(result.items())}


def _ontology_maps(ontology: object) -> tuple[dict[str, str], frozenset[str]]:
    definitions = getattr(ontology, "entity_types", None)
    if not isinstance(definitions, Mapping) or not definitions:
        raise ValueError("ontology must expose nonempty entity_types")
    canonical: dict[str, str] = {}
    for raw_name, definition in sorted(definitions.items()):
        name = _bounded_text(raw_name, "ontology entity type", maximum=128)
        canonical_name = _bounded_text(
            getattr(definition, "name", name), "ontology entity type", maximum=128
        )
        for candidate in (name, *getattr(definition, "aliases", ())):
            alias = _bounded_text(candidate, "ontology type alias", maximum=128)
            existing = canonical.setdefault(alias, canonical_name)
            if existing != canonical_name:
                raise ValueError("ontology entity type aliases are ambiguous")
    relations = getattr(ontology, "relations", {})
    if not isinstance(relations, Mapping):
        raise ValueError("ontology relations must be a mapping")
    return canonical, frozenset(
        _bounded_text(name, "ontology relation", maximum=128) for name in relations
    )


def _pair(left: int, right: int) -> tuple[int, int]:
    if left == right:
        raise ValueError("identity pair endpoints must differ")
    return (left, right) if left < right else (right, left)


def _star_pairs(values: Sequence[int]) -> tuple[tuple[int, int], ...]:
    """Represent one equivalence block with linear deterministic evidence edges."""

    ordered = tuple(sorted(values))
    if len(ordered) < 2:
        return ()
    return tuple(_pair(ordered[0], value) for value in ordered[1:])


def _label_keys(entity: DocumentEntityInput) -> frozenset[str]:
    return frozenset(
        normalize_entity_label(value).key
        for value in (
            entity.label,
            entity.normalized_label,
            *(alias.alias for alias in entity.alias_evidence),
        )
    )


def _is_undefined_acronym(entity: DocumentEntityInput) -> bool:
    return bool(
        _ACRONYM_PATTERN.fullmatch(entity.label)
        and not entity.identifier
        and not any(
            alias.method == "defined_acronym" for alias in entity.alias_evidence
        )
    )


def _conflict_reason(
    left: DocumentEntityInput, right: DocumentEntityInput
) -> str | None:
    if left.version_signature != right.version_signature and (
        left.version_signature or right.version_signature
    ):
        return "version_signature_conflict"
    if left.identifier and right.identifier and left.identifier != right.identifier:
        return "conflicting_stable_identifiers"
    return None


def _decision(
    left: int,
    right: int,
    *,
    outcome: ResolutionOutcome,
    tier: ResolutionTier,
    confidence: float,
    reason_codes: tuple[str, ...],
    identifier_score: float | None = None,
    alias_score: float | None = None,
    embedding_similarity: float | None = None,
    neighborhood_agreement: float | None = None,
    signature: str | None = None,
    left_hash: str | None = None,
    right_hash: str | None = None,
    rank: int | None = None,
) -> CollectionPairDecision:
    left, right = _pair(left, right)
    return CollectionPairDecision(
        left_entity_id=left,
        right_entity_id=right,
        outcome=outcome,
        tier=tier,
        resolution_confidence=_quantize(confidence),  # type: ignore[arg-type]
        identifier_score=_quantize(identifier_score),
        alias_score=_quantize(alias_score),
        embedding_similarity=_quantize(embedding_similarity),
        neighborhood_agreement=_quantize(neighborhood_agreement),
        reason_codes=tuple(sorted(reason_codes)),
        embedding_model_signature=signature,
        left_embedding_input_hash=left_hash,
        right_embedding_input_hash=right_hash,
        candidate_rank=rank,
    )


def _identity_signature(
    entity: DocumentEntityInput, canonical_type: str | None
) -> tuple[str, str, str, str]:
    return (
        canonical_type or f"unknown:{entity.entity_type}",
        entity.identifier,
        normalize_entity_label(entity.normalized_label).key,
        entity.version_signature,
    )


def _precomputed_relation_signatures(
    relations: Sequence[SupportedRelation],
    *,
    entities: Mapping[int, DocumentEntityInput],
    canonical_types: Mapping[int, str | None],
    allowed_relations: frozenset[str],
    threshold: float,
) -> dict[int, frozenset[tuple[object, ...]]]:
    """Freeze neighborhood evidence before any identity union can affect it."""

    signatures: dict[int, set[tuple[object, ...]]] = defaultdict(set)
    seen: set[int] = set()
    for relation in sorted(relations, key=lambda item: item.relation_id):
        if relation.relation_id in seen:
            raise ValueError("duplicate supported relation id")
        seen.add(relation.relation_id)
        if (
            relation.source_entity_id not in entities
            or relation.target_entity_id not in entities
        ):
            raise ValueError("supported relation endpoint is outside snapshot")
        if (
            not relation.supported
            or relation.confidence < threshold
            or relation.relation_type not in allowed_relations
        ):
            continue
        source = entities[relation.source_entity_id]
        target = entities[relation.target_entity_id]
        signatures[source.entity_id].add(
            (
                "out",
                relation.relation_type,
                *_identity_signature(target, canonical_types[target.entity_id]),
            )
        )
        signatures[target.entity_id].add(
            (
                "in",
                relation.relation_type,
                *_identity_signature(source, canonical_types[source.entity_id]),
            )
        )
    return {key: frozenset(value) for key, value in signatures.items()}


def _jaccard(left: frozenset[object], right: frozenset[object]) -> float:
    if not left or not right:
        return 0.0
    return len(left.intersection(right)) / len(left.union(right))


def _representative(
    members: Sequence[DocumentEntityInput],
) -> DocumentEntityInput:
    return min(
        members,
        key=lambda entity: (
            -entity.extraction_confidence,
            -entity.document_resolution_confidence,
            len(entity.label),
            entity.label.casefold(),
            entity.label,
            entity.entity_id,
        ),
    )


def _embedding_text(members: Sequence[DocumentEntityInput]) -> str:
    representative = _representative(members)
    descriptions = sorted({item.description for item in members if item.description})
    if not descriptions:
        return representative.label
    return f"{representative.label}\n{' '.join(descriptions)}"[:MAX_TEXT_CHARACTERS]


def _lexical_block_keys(members: Sequence[DocumentEntityInput]) -> frozenset[str]:
    values = {
        normalize_entity_label(value).key
        for member in members
        for value in (member.label, *(alias.alias for alias in member.alias_evidence))
    }
    keys: set[str] = set()
    for value in values:
        tokens = tuple(token for token in value.split() if len(token) >= 3)
        keys.update(f"token:{token}" for token in tokens)
        compact = "".join(character for character in value if character.isalnum())
        keys.update(
            f"tri:{compact[index : index + 3]}"
            for index in range(max(0, len(compact) - 2))
        )
    return frozenset(keys)


def _source_entity_content(entity: DocumentEntityInput) -> dict[str, object]:
    return {
        "entity_id": entity.entity_id,
        "document_cluster_key": entity.document_cluster_key,
        "document_artifact_id": entity.document_artifact_id,
        "document_id": str(entity.document_id),
        "label": entity.label,
        "normalized_label": entity.normalized_label,
        "entity_type": entity.entity_type,
        "identifier": entity.identifier,
        "version_signature": entity.version_signature,
        "alias_evidence": [
            {
                "alias": alias.alias,
                "method": alias.method,
                "mention_id": alias.mention_id,
            }
            for alias in entity.alias_evidence
        ],
        "description": entity.description,
        "extraction_confidence": entity.extraction_confidence,
        "document_resolution_confidence": entity.document_resolution_confidence,
    }


def _source_entity_fingerprint(entities: Sequence[DocumentEntityInput]) -> str:
    return _hash([_source_entity_content(entity) for entity in entities])


def _source_relation_fingerprint(relations: Sequence[SupportedRelation]) -> str:
    return _hash(
        [
            {
                "relation_id": relation.relation_id,
                "source_entity_id": relation.source_entity_id,
                "relation_type": relation.relation_type,
                "target_entity_id": relation.target_entity_id,
                "confidence": relation.confidence,
                "supported": relation.supported,
            }
            for relation in relations
        ]
    )


def _input_fingerprint(
    snapshot: CollectionBuildSnapshot,
    entities: Sequence[DocumentEntityInput],
    relations: Sequence[SupportedRelation],
    config: CollectionResolutionConfig,
    embedding_signature: str,
) -> str:
    return _hash(
        {
            "snapshot": _snapshot_content(snapshot),
            "entities": [_source_entity_content(entity) for entity in entities],
            "relations": [
                {
                    "relation_id": relation.relation_id,
                    "source_entity_id": relation.source_entity_id,
                    "relation_type": relation.relation_type,
                    "target_entity_id": relation.target_entity_id,
                    "confidence": relation.confidence,
                    "supported": relation.supported,
                }
                for relation in relations
            ],
            "config": {
                "automatic": config.thresholds.automatic,
                "candidate": config.thresholds.candidate,
                "retrieval_similarity": config.thresholds.retrieval_similarity,
                "max_candidates_per_entity": config.max_candidates_per_entity,
                "max_candidate_pool_per_entity": config.max_candidate_pool_per_entity,
                "exact_semantic_scan_limit": config.exact_semantic_scan_limit,
                "embedding_weight": config.embedding_weight,
                "neighborhood_weight": config.neighborhood_weight,
                "relation_support_threshold": config.relation_support_threshold,
                "max_entities": config.max_entities,
            },
            "embedding_model_signature": embedding_signature,
        }
    )


def _snapshot_content(snapshot: CollectionBuildSnapshot) -> dict[str, object]:
    return {
        "destination_artifact_id": snapshot.destination_artifact_id,
        "collection_id": snapshot.collection_id,
        "inputs": [
            {
                "manifest_input_id": item.manifest_input_id,
                "document_artifact_id": item.document_artifact_id,
                "document_id": str(item.document_id),
                "source_signature": item.source_signature,
                "build_signature": item.build_signature,
            }
            for item in snapshot.inputs
        ],
        "source_hash": snapshot.source_hash,
        "ontology_checksum": snapshot.ontology_checksum,
    }


def _result_content(result: CollectionResolutionResult) -> dict[str, object]:
    return {
        "snapshot": _snapshot_content(result.snapshot),
        "resolver_version": result.resolver_version,
        "input_fingerprint": result.input_fingerprint,
        "source_entity_fingerprint": result.source_entity_fingerprint,
        "source_relation_fingerprint": result.source_relation_fingerprint,
        "source_entity_ids": list(result.source_entity_ids),
        "clusters": [
            {
                "cluster_key": cluster.cluster_key,
                "document_entity_ids": list(cluster.document_entity_ids),
                "document_cluster_keys": list(cluster.document_cluster_keys),
                "document_artifact_ids": list(cluster.document_artifact_ids),
                "document_ids": [str(value) for value in cluster.document_ids],
                "label": cluster.label,
                "normalized_label": cluster.normalized_label,
                "version_signature": cluster.version_signature,
                "entity_type": cluster.entity_type,
                "identifier": cluster.identifier,
                "aliases": list(cluster.aliases),
                "embedding_hash": (
                    None
                    if cluster.embedding is None
                    else _hash(list(cluster.embedding))
                ),
                "embedding_model_signature": cluster.embedding_model_signature,
                "embedding_input_hash": cluster.embedding_input_hash,
                "extraction_confidence": cluster.extraction_confidence,
                "resolution_confidence": cluster.resolution_confidence,
                "retrieval_utility": cluster.retrieval_utility,
                "promotion_confidence": cluster.promotion_confidence,
                "resolution_methods": list(cluster.resolution_methods),
            }
            for cluster in result.clusters
        ],
        "decisions": [
            {
                "left_entity_id": item.left_entity_id,
                "right_entity_id": item.right_entity_id,
                "outcome": item.outcome.value,
                "tier": item.tier.value,
                "resolution_confidence": item.resolution_confidence,
                "identifier_score": item.identifier_score,
                "alias_score": item.alias_score,
                "embedding_similarity": item.embedding_similarity,
                "neighborhood_agreement": item.neighborhood_agreement,
                "reason_codes": list(item.reason_codes),
                "embedding_model_signature": item.embedding_model_signature,
                "left_embedding_input_hash": item.left_embedding_input_hash,
                "right_embedding_input_hash": item.right_embedding_input_hash,
                "candidate_rank": item.candidate_rank,
            }
            for item in result.decisions
        ],
        "audit": {
            name: getattr(result.audit, name)
            for name in result.audit.__dataclass_fields__
        },
    }


def resolution_result_checksum(result: CollectionResolutionResult) -> str:
    return _hash(_result_content(result))


def _component_anchors(
    members: Sequence[int], entities: Mapping[int, DocumentEntityInput]
) -> tuple[frozenset[str], frozenset[str]]:
    identifiers = frozenset(
        entities[item].identifier for item in members if entities[item].identifier
    )
    versions = frozenset(
        entities[item].version_signature
        for item in members
        if entities[item].version_signature
    )
    return identifiers, versions


def _semantic_pair_is_eligible(
    left: int,
    right: int,
    *,
    deterministic_groups: Mapping[int, tuple[int, ...]],
    entities: Mapping[int, DocumentEntityInput],
) -> bool:
    left_members = tuple(entities[item] for item in deterministic_groups[left])
    right_members = tuple(entities[item] for item in deterministic_groups[right])
    left_ids, left_versions = _component_anchors(deterministic_groups[left], entities)
    right_ids, right_versions = _component_anchors(
        deterministic_groups[right], entities
    )
    if (left_ids and right_ids and left_ids != right_ids) or (
        left_versions != right_versions and (left_versions or right_versions)
    ):
        return False
    if all(
        _is_undefined_acronym(item) and not item.description for item in left_members
    ):
        return False
    if all(
        _is_undefined_acronym(item) and not item.description for item in right_members
    ):
        return False
    return True


def _component_stable_key(
    members: Sequence[int], entities: Mapping[int, DocumentEntityInput]
) -> str:
    """Address a deterministic component without using database insertion IDs."""

    return _hash(sorted(entities[item].document_cluster_key for item in members))


def _component_pair_stable_key(
    left: int,
    right: int,
    component_keys: Mapping[int, str],
) -> tuple[str, str]:
    return tuple(sorted((component_keys[left], component_keys[right])))  # type: ignore[return-value]


def _automatic_conflict_components(
    roots: Sequence[int],
    automatic_edges: Sequence[tuple[int, int]],
    deterministic_groups: Mapping[int, tuple[int, ...]],
    entities: Mapping[int, DocumentEntityInput],
) -> frozenset[tuple[int, int]]:
    graph = _DisjointSet(roots)
    for left, right in automatic_edges:
        graph.union(left, right)
    conflicting: set[tuple[int, int]] = set()
    for component in graph.groups().values():
        identifiers: set[str] = set()
        versions: set[str] = set()
        for root in component:
            component_identifiers, component_versions = _component_anchors(
                deterministic_groups[root], entities
            )
            identifiers.update(component_identifiers)
            versions.update(component_versions)
        if len(identifiers) > 1 or len(versions) > 1:
            component_set = set(component)
            conflicting.update(
                _pair(left, right)
                for left, right in automatic_edges
                if left in component_set and right in component_set
            )
    return frozenset(conflicting)


def resolve_collection_entities(
    snapshot: CollectionBuildSnapshot,
    entities: Iterable[DocumentEntityInput],
    ontology: object,
    *,
    relations: Iterable[SupportedRelation] = (),
    config: CollectionResolutionConfig | None = None,
    embedding_session: CollectionEmbeddingSession,
) -> CollectionResolutionResult:
    """Resolve document nodes without reading outside ``snapshot`` or using chat."""

    if type(snapshot) is not CollectionBuildSnapshot:
        raise ValueError("snapshot must be exact CollectionBuildSnapshot")
    snapshot.__post_init__()
    if type(embedding_session) is not CollectionEmbeddingSession:
        raise ValueError("resolution requires a build-scoped embedding session")
    config = CollectionResolutionConfig() if config is None else config
    if type(config) is not CollectionResolutionConfig:
        raise ValueError("config must be exact CollectionResolutionConfig")
    config.__post_init__()
    if getattr(ontology, "checksum", None) != snapshot.ontology_checksum:
        raise ValueError("ontology checksum does not match snapshot")
    canonical_map, allowed_relations = _ontology_maps(ontology)

    inputs = tuple(entities)
    if len(inputs) > config.max_entities:
        raise ValueError("collection entity input exceeds configured limit")
    if any(type(item) is not DocumentEntityInput for item in inputs):
        raise ValueError("entities must contain exact DocumentEntityInput values")
    for item in inputs:
        item.__post_init__()
    ordered = tuple(sorted(inputs, key=lambda item: item.entity_id))
    entity_ids = tuple(item.entity_id for item in ordered)
    if len(entity_ids) != len(set(entity_ids)):
        raise ValueError("duplicate document entity id")
    snapshot_documents = {
        item.document_artifact_id: item.document_id for item in snapshot.inputs
    }
    if any(
        snapshot_documents.get(item.document_artifact_id) != item.document_id
        for item in ordered
    ):
        raise ValueError("document entity is outside exact manifest snapshot")

    relation_inputs = tuple(relations)
    if len(relation_inputs) > MAX_RELATIONS:
        raise ValueError("relation input exceeds configured limit")
    if any(type(item) is not SupportedRelation for item in relation_inputs):
        raise ValueError("relations must contain exact SupportedRelation values")
    for item in relation_inputs:
        item.__post_init__()
    ordered_relations = tuple(
        sorted(relation_inputs, key=lambda item: item.relation_id)
    )

    by_id = {item.entity_id: item for item in ordered}
    canonical_type = {
        item.entity_id: canonical_map.get(item.entity_type) for item in ordered
    }
    relation_signatures = _precomputed_relation_signatures(
        ordered_relations,
        entities=by_id,
        canonical_types=canonical_type,
        allowed_relations=allowed_relations,
        threshold=config.relation_support_threshold,
    )

    counts_by_type: dict[str, int] = defaultdict(int)
    for item in ordered:
        if canonical_type[item.entity_id] is not None:
            counts_by_type[canonical_type[item.entity_id]] += 1  # type: ignore[index]
    total_pairs = len(ordered) * (len(ordered) - 1) // 2
    compatible_pairs = sum(
        count * (count - 1) // 2 for count in counts_by_type.values()
    )
    incompatible_pair_count = total_pairs - compatible_pairs

    dsu = _DisjointSet(entity_ids)
    decisions: dict[tuple[int, int], CollectionPairDecision] = {}
    identifier_blocks: dict[tuple[str, str], list[int]] = defaultdict(list)
    for item in ordered:
        item_type = canonical_type[item.entity_id]
        if item_type is not None and item.identifier:
            identifier_blocks[(item_type, item.identifier)].append(item.entity_id)
    for _block_key, members in sorted(identifier_blocks.items()):
        by_version: dict[str, list[int]] = defaultdict(list)
        for member in members:
            by_version[by_id[member].version_signature].append(member)
        representatives: list[int] = []
        for _version, version_members in sorted(by_version.items()):
            representatives.append(min(version_members))
            for left, right in _star_pairs(version_members):
                decisions[(left, right)] = _decision(
                    left,
                    right,
                    outcome=ResolutionOutcome.AUTOMATIC,
                    tier=ResolutionTier.STABLE_IDENTIFIER,
                    confidence=1.0,
                    identifier_score=1.0,
                    reason_codes=("stable_identifier_equal",),
                )
                dsu.union(left, right)
        for left, right in _star_pairs(representatives):
            decisions[(left, right)] = _decision(
                left,
                right,
                outcome=ResolutionOutcome.REJECTED,
                tier=ResolutionTier.STABLE_IDENTIFIER,
                confidence=0.0,
                identifier_score=1.0,
                reason_codes=("version_signature_conflict",),
            )

    label_blocks: dict[tuple[str, str], set[int]] = defaultdict(set)
    for item in ordered:
        item_type = canonical_type[item.entity_id]
        if item_type is not None:
            for key in _label_keys(item):
                label_blocks[(item_type, key)].add(item.entity_id)
    for _block_key, members in sorted(label_blocks.items()):
        ordered_members = tuple(sorted(members))
        anchors = {
            by_id[item].identifier for item in ordered_members if by_id[item].identifier
        }
        if len(anchors) > 1:
            for left, right in _star_pairs(ordered_members):
                pair = _pair(left, right)
                if pair in decisions:
                    continue
                conflict = _conflict_reason(by_id[left], by_id[right])
                decisions[pair] = _decision(
                    left,
                    right,
                    outcome=ResolutionOutcome.REJECTED,
                    tier=ResolutionTier.EXACT_LABEL_OR_ALIAS,
                    confidence=0.0,
                    alias_score=1.0,
                    reason_codes=(conflict or "ambiguous_label_identifier_block",),
                )
            continue

        categories: dict[tuple[str, bool], list[int]] = defaultdict(list)
        for member in ordered_members:
            categories[
                (
                    by_id[member].version_signature,
                    _is_undefined_acronym(by_id[member]),
                )
            ].append(member)
        representatives: list[int] = []
        for (_version, undefined), category_members in sorted(categories.items()):
            representatives.append(min(category_members))
            for left, right in _star_pairs(category_members):
                pair = _pair(left, right)
                if pair in decisions:
                    continue
                if undefined:
                    decisions[pair] = _decision(
                        left,
                        right,
                        outcome=ResolutionOutcome.REJECTED,
                        tier=ResolutionTier.EXACT_LABEL_OR_ALIAS,
                        confidence=0.0,
                        alias_score=1.0,
                        reason_codes=("undefined_acronym",),
                    )
                else:
                    decisions[pair] = _decision(
                        left,
                        right,
                        outcome=ResolutionOutcome.AUTOMATIC,
                        tier=ResolutionTier.EXACT_LABEL_OR_ALIAS,
                        confidence=0.99,
                        alias_score=1.0,
                        reason_codes=("exact_evidence_backed_label_or_alias",),
                    )
                    dsu.union(left, right)
        for left, right in _star_pairs(representatives):
            pair = _pair(left, right)
            if pair in decisions:
                continue
            conflict = _conflict_reason(by_id[left], by_id[right])
            decisions[pair] = _decision(
                left,
                right,
                outcome=ResolutionOutcome.REJECTED,
                tier=ResolutionTier.EXACT_LABEL_OR_ALIAS,
                confidence=0.0,
                alias_score=1.0,
                reason_codes=(conflict or "undefined_acronym",),
            )

    deterministic_groups = dsu.groups()
    component_keys = {
        root: _component_stable_key(members, by_id)
        for root, members in deterministic_groups.items()
    }
    roots_by_type: dict[str, list[int]] = defaultdict(list)
    group_members: dict[int, tuple[DocumentEntityInput, ...]] = {}
    for root, members in deterministic_groups.items():
        group_members[root] = tuple(by_id[item] for item in members)
        group_type = canonical_type[members[0]]
        if group_type is not None:
            roots_by_type[group_type].append(root)

    pair_pool: set[tuple[int, int]] = set()
    for roots in roots_by_type.values():
        roots = sorted(roots)
        if len(roots) < 2:
            continue
        blocks: dict[str, list[int]] = defaultdict(list)
        for root in roots:
            for key in sorted(_lexical_block_keys(group_members[root]))[:128]:
                blocks[key].append(root)
        local: dict[int, set[int]] = defaultdict(set)
        if len(roots) <= config.exact_semantic_scan_limit:
            for index, left in enumerate(roots):
                local[left].update(roots[index + 1 :])
                for right in roots[:index]:
                    local[left].add(right)
        else:
            for block_roots in blocks.values():
                block_roots = sorted(
                    set(block_roots),
                    key=lambda root: component_keys[root],
                )
                if len(block_roots) < 2:
                    continue
                limit = min(
                    config.max_candidate_pool_per_entity,
                    len(block_roots) - 1,
                )
                for index, left in enumerate(block_roots):
                    for offset in range(1, limit + 1):
                        if len(local[left]) >= config.max_candidate_pool_per_entity:
                            break
                        local[left].add(
                            block_roots[(index + offset) % len(block_roots)]
                        )
        for left, candidates in local.items():
            bounded = sorted(
                candidates,
                key=lambda right: (
                    _hash(_component_pair_stable_key(left, right, component_keys)),
                    component_keys[right],
                ),
            )[: config.max_candidate_pool_per_entity]
            pair_pool.update(
                _pair(left, right)
                for right in bounded
                if _semantic_pair_is_eligible(
                    left,
                    right,
                    deterministic_groups=deterministic_groups,
                    entities=by_id,
                )
            )

    roots_to_embed = tuple(sorted({root for pair in pair_pool for root in pair}))
    embedded_by_root: dict[int, EmbeddedText] = {}
    if roots_to_embed:
        texts = tuple(_embedding_text(group_members[root]) for root in roots_to_embed)
        embedded = embedding_session.embed(texts)
        embedded_by_root = dict(zip(roots_to_embed, embedded, strict=True))

    scored: list[tuple[float, int, int, float, float, ResolutionTier]] = []
    for left, right in sorted(pair_pool):
        left_ids, left_versions = _component_anchors(deterministic_groups[left], by_id)
        right_ids, right_versions = _component_anchors(
            deterministic_groups[right], by_id
        )
        if (left_ids and right_ids and left_ids != right_ids) or (
            left_versions and right_versions and left_versions != right_versions
        ):
            continue
        similarity = cosine_similarity(
            embedded_by_root[left].vector, embedded_by_root[right].vector
        )
        left_neighborhood = frozenset(
            signature
            for item in deterministic_groups[left]
            for signature in relation_signatures.get(item, ())
        )
        right_neighborhood = frozenset(
            signature
            for item in deterministic_groups[right]
            for signature in relation_signatures.get(item, ())
        )
        neighborhood = _jaccard(left_neighborhood, right_neighborhood)
        if neighborhood:
            combined = combine_resolution_scores(
                embedding_similarity=similarity,
                neighborhood_agreement=neighborhood,
                embedding_weight=config.embedding_weight,
                neighborhood_weight=config.neighborhood_weight,
            ).composite
            tier = ResolutionTier.NEIGHBORHOOD_AGREEMENT
        else:
            combined = similarity
            tier = ResolutionTier.EMBEDDING
        scored.append(
            (
                _quantize(combined),
                left,
                right,
                similarity,
                neighborhood,
                tier,
            )
        )

    fanout: dict[int, int] = defaultdict(int)
    selected: list[tuple[float, int, int, float, float, ResolutionTier, int]] = []
    for score, left, right, similarity, neighborhood, tier in sorted(
        scored,
        key=lambda item: (
            -item[0],
            _component_pair_stable_key(item[1], item[2], component_keys),
        ),
    ):
        if (
            fanout[left] >= config.max_candidates_per_entity
            or fanout[right] >= config.max_candidates_per_entity
        ):
            continue
        fanout[left] += 1
        fanout[right] += 1
        rank = max(fanout[left], fanout[right])
        selected.append((score, left, right, similarity, neighborhood, tier, rank))

    provisional_auto = [
        (left, right)
        for score, left, right, _similarity, _neighbor, _tier, _rank in selected
        if classify_resolution_score(score, config.thresholds)
        is ResolutionOutcome.AUTOMATIC
    ]
    conflict_edges = _automatic_conflict_components(
        tuple(deterministic_groups),
        provisional_auto,
        deterministic_groups,
        by_id,
    )
    for score, left_root, right_root, similarity, neighborhood, tier, rank in selected:
        representative = min(
            _pair(left, right)
            for left in deterministic_groups[left_root]
            for right in deterministic_groups[right_root]
        )
        outcome = classify_resolution_score(score, config.thresholds)
        reasons: tuple[str, ...]
        if (
            outcome is ResolutionOutcome.AUTOMATIC
            and _pair(left_root, right_root) in conflict_edges
        ):
            outcome = ResolutionOutcome.CANDIDATE
            reasons = ("transitive_identity_conflict",)
        else:
            reasons = (
                {
                    ResolutionOutcome.AUTOMATIC: "automatic_threshold_met",
                    ResolutionOutcome.CANDIDATE: "candidate_review_threshold_met",
                    ResolutionOutcome.REJECTED: "below_candidate_threshold",
                }[outcome],
            )
        left_embedding = embedded_by_root[left_root]
        right_embedding = embedded_by_root[right_root]
        decisions[representative] = _decision(
            *representative,
            outcome=outcome,
            tier=tier,
            confidence=score,
            embedding_similarity=similarity,
            neighborhood_agreement=neighborhood,
            reason_codes=reasons,
            signature=embedding_session.expected_model_signature,
            left_hash=left_embedding.input_hash,
            right_hash=right_embedding.input_hash,
            rank=rank,
        )
        if outcome is ResolutionOutcome.AUTOMATIC:
            dsu.union(left_root, right_root)

    final_groups = dsu.groups()
    clusters: list[CollectionEntityCluster] = []
    for _root, member_ids in final_groups.items():
        members = tuple(by_id[item] for item in member_ids)
        representative = _representative(members)
        accepted = tuple(
            item
            for item in decisions.values()
            if item.outcome is ResolutionOutcome.AUTOMATIC
            and item.left_entity_id in member_ids
            and item.right_entity_id in member_ids
        )
        identifiers = sorted({item.identifier for item in members if item.identifier})
        versions = sorted(
            {item.version_signature for item in members if item.version_signature}
        )
        contributing = tuple(
            embedded_by_root[root]
            for root, deterministic_members in deterministic_groups.items()
            if root in embedded_by_root
            and set(deterministic_members).intersection(member_ids)
        )
        embedding: tuple[float, ...] | None = None
        embedding_input_hash: str | None = None
        if contributing:
            embedding = tuple(
                sum(item.vector[index] for item in contributing) / len(contributing)
                for index in range(EMBEDDING_DIMENSIONS)
            )
            embedding_input_hash = _hash(
                sorted(item.input_hash for item in contributing)
            )
        aliases = tuple(
            sorted(
                {
                    evidence.alias
                    for member in members
                    for evidence in member.alias_evidence
                },
                key=lambda value: (value.casefold(), value),
            )
        )
        entity_type = (
            canonical_type[representative.entity_id] or representative.entity_type
        )
        document_cluster_keys = tuple(
            sorted(item.document_cluster_key for item in members)
        )
        cluster_key = _hash(
            {
                "resolver_version": COLLECTION_RESOLVER_VERSION,
                "ontology_checksum": snapshot.ontology_checksum,
                "document_cluster_keys": list(document_cluster_keys),
                "entity_type": entity_type,
                "identifier": identifiers[0] if len(identifiers) == 1 else "",
                "version_signature": versions[0] if len(versions) == 1 else "",
            }
        )
        clusters.append(
            CollectionEntityCluster(
                cluster_key=cluster_key,
                document_entity_ids=member_ids,
                document_cluster_keys=document_cluster_keys,
                document_artifact_ids=tuple(
                    sorted({item.document_artifact_id for item in members})
                ),
                document_ids=tuple(
                    sorted({item.document_id for item in members}, key=str)
                ),
                label=representative.label,
                normalized_label=normalize_entity_label(
                    representative.normalized_label
                ).key,
                version_signature=versions[0] if len(versions) == 1 else "",
                entity_type=entity_type,
                identifier=identifiers[0] if len(identifiers) == 1 else "",
                aliases=aliases,
                embedding=embedding,
                embedding_model_signature=(
                    embedding_session.expected_model_signature if embedding else None
                ),
                embedding_input_hash=embedding_input_hash,
                extraction_confidence=_quantize(
                    sum(item.extraction_confidence for item in members) / len(members)
                ),  # type: ignore[arg-type]
                resolution_confidence=_quantize(
                    min(
                        [item.document_resolution_confidence for item in members]
                        + [item.resolution_confidence for item in accepted]
                    )
                ),  # type: ignore[arg-type]
                retrieval_utility=None,
                promotion_confidence=None,
                resolution_methods=tuple(sorted({item.tier.value for item in accepted}))
                or ("singleton",),
            )
        )
    clusters.sort(key=lambda item: item.cluster_key)
    ordered_decisions = tuple(decisions[key] for key in sorted(decisions))
    audit = ResolutionAudit(
        embedding_model_signature=embedding_session.expected_model_signature,
        max_candidates_per_entity=config.max_candidates_per_entity,
        max_candidate_pool_per_entity=config.max_candidate_pool_per_entity,
        max_observed_candidate_fanout=max(fanout.values(), default=0),
        embedded_entity_count=len(roots_to_embed),
        embedding_candidate_pair_count=len(selected),
        type_incompatible_pair_count=incompatible_pair_count,
        automatic_pair_count=sum(
            item.outcome is ResolutionOutcome.AUTOMATIC for item in ordered_decisions
        ),
        candidate_pair_count=sum(
            item.outcome is ResolutionOutcome.CANDIDATE for item in ordered_decisions
        ),
        rejected_pair_count=sum(
            item.outcome is ResolutionOutcome.REJECTED for item in ordered_decisions
        ),
    )
    result = CollectionResolutionResult(
        snapshot=snapshot,
        resolver_version=COLLECTION_RESOLVER_VERSION,
        input_fingerprint=_input_fingerprint(
            snapshot,
            ordered,
            ordered_relations,
            config,
            embedding_session.expected_model_signature,
        ),
        source_entity_fingerprint=_source_entity_fingerprint(ordered),
        source_relation_fingerprint=_source_relation_fingerprint(ordered_relations),
        source_entity_ids=entity_ids,
        clusters=tuple(clusters),
        decisions=ordered_decisions,
        audit=audit,
        checksum="",
    )
    return CollectionResolutionResult(
        snapshot=result.snapshot,
        resolver_version=result.resolver_version,
        input_fingerprint=result.input_fingerprint,
        source_entity_fingerprint=result.source_entity_fingerprint,
        source_relation_fingerprint=result.source_relation_fingerprint,
        source_entity_ids=result.source_entity_ids,
        clusters=result.clusters,
        decisions=result.decisions,
        audit=result.audit,
        checksum=resolution_result_checksum(result),
    )


class CollectionResolutionPersistenceError(RuntimeError):
    """Raised when a collection resolution write cannot preserve its snapshot."""


def build_collection_snapshot(
    *,
    collection,
    document_artifacts: Iterable[object],
    ontology_version: str,
    ontology_checksum: str,
    extractor_version: str,
    resolver_version: str,
    filter_policy_version: str,
    embedding_model_signature: str,
):
    """Create a building collection artifact and its immutable source manifest."""

    from django.db import transaction

    from apps.collections.models import Collection
    from apps.documents.models import Document
    from apps.knowledge_graph.models import (
        CollectionArtifactInput,
        GraphArtifact,
    )
    from apps.knowledge_graph.models.inputs import (
        collection_input_source_signature,
        collection_manifest_source_hash,
    )

    if type(collection) is not Collection or collection.pk is None:
        raise CollectionResolutionPersistenceError(
            "collection snapshot requires a persisted exact Collection"
        )
    _require_hash(ontology_checksum, "ontology checksum")
    expected_signature = _bounded_text(
        embedding_model_signature, "embedding model signature", maximum=512
    )
    if "dims=1024" not in expected_signature or "prep=" not in expected_signature:
        raise CollectionResolutionPersistenceError(
            "collection embedding signature must lock dimensions and preprocessing"
        )
    source_objects = tuple(document_artifacts)
    source_ids = tuple(getattr(item, "pk", None) for item in source_objects)
    if any(type(item) is not int or item <= 0 for item in source_ids):
        raise CollectionResolutionPersistenceError(
            "document artifacts must be persisted GraphArtifact rows"
        )
    if len(source_ids) != len(set(source_ids)):
        raise CollectionResolutionPersistenceError(
            "document artifact snapshot contains duplicates"
        )
    with transaction.atomic():
        sources = tuple(
            GraphArtifact.objects.select_for_update()
            .filter(pk__in=source_ids)
            .order_by("pk")
        )
        if tuple(item.pk for item in sources) != tuple(sorted(source_ids)):
            raise CollectionResolutionPersistenceError(
                "document artifact snapshot changed before locking"
            )
        source_rows: list[tuple[GraphArtifact, uuid.UUID, str]] = []
        for source in sources:
            if (
                source.scope_type != GraphArtifact.ScopeType.DOCUMENT
                or source.status != GraphArtifact.Status.ACTIVE
            ):
                raise CollectionResolutionPersistenceError(
                    "collection inputs must be active document artifacts"
                )
            try:
                document_id = uuid.UUID(source.scope_id)
            except ValueError as exc:
                raise CollectionResolutionPersistenceError(
                    "document artifact scope is not canonical"
                ) from exc
            document = Document.get_by_id(document_id)
            if document is None or document.collection_id != collection.pk:
                raise CollectionResolutionPersistenceError(
                    "manifest document does not belong to the collection"
                )
            if source.ontology_version != ontology_version:
                raise CollectionResolutionPersistenceError(
                    "manifest document ontology does not match collection build"
                )
            source_signature = collection_input_source_signature(
                collection_id=collection.pk,
                document_id=document_id,
                document_artifact=source,
            )
            source_rows.append((source, document_id, source_signature))
        source_hash = collection_manifest_source_hash(row[2] for row in source_rows)
        artifact = GraphArtifact.objects.create(
            scope_type=GraphArtifact.ScopeType.COLLECTION,
            scope_id=collection.pk,
            status=GraphArtifact.Status.BUILDING,
            source_hash=source_hash,
            ontology_version=_bounded_text(
                ontology_version, "ontology version", maximum=128
            ),
            extractor_version=_bounded_text(
                extractor_version, "extractor version", maximum=128
            ),
            resolver_version=_bounded_text(
                resolver_version, "resolver version", maximum=128
            ),
            filter_policy_version=_bounded_text(
                filter_policy_version, "filter policy version", maximum=128
            ),
            embedding_model_signature=expected_signature,
            metadata={
                "ontology_checksum": ontology_checksum,
                "manifest_version": 1,
            },
        )
        rows = [
            CollectionArtifactInput(
                artifact=artifact,
                collection=collection,
                document_id=document_id,
                document_artifact=source,
                source_signature=source_signature,
                build_signature="0" * 64,
            )
            for source, document_id, source_signature in source_rows
        ]
        CollectionArtifactInput.objects.bulk_create(rows)
        return artifact, tuple(rows)


def _validate_collection_destination(artifact, run) -> None:
    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun

    if artifact.scope_type != GraphArtifact.ScopeType.COLLECTION:
        raise CollectionResolutionPersistenceError(
            "collection resolution requires a collection artifact"
        )
    if artifact.status != GraphArtifact.Status.BUILDING:
        raise CollectionResolutionPersistenceError(
            "collection resolution destination must be building"
        )
    if run.artifact_id != artifact.pk:
        raise CollectionResolutionPersistenceError(
            "build run must belong to destination artifact"
        )
    if run.status != GraphBuildRun.Status.RUNNING:
        raise CollectionResolutionPersistenceError("build run must be running")
    if run.stage != GraphBuildRun.Stage.RESOLUTION:
        raise CollectionResolutionPersistenceError(
            "collection resolver requires the resolution stage"
        )
    for field in (
        "scope_type",
        "scope_id",
        "source_hash",
        "ontology_version",
        "extractor_version",
        "resolver_version",
        "filter_policy_version",
        "embedding_model_signature",
    ):
        if getattr(run, field) != getattr(artifact, field):
            raise CollectionResolutionPersistenceError(
                f"build run {field} does not match destination artifact"
            )


def _snapshot_from_locked_manifest(artifact, manifest_rows) -> CollectionBuildSnapshot:
    from apps.knowledge_graph.models import GraphArtifact
    from apps.knowledge_graph.models.inputs import (
        collection_input_build_signature,
        collection_input_source_signature,
        collection_manifest_source_hash,
    )

    if artifact.scope_type != GraphArtifact.ScopeType.COLLECTION:
        raise CollectionResolutionPersistenceError("manifest owner is not a collection")
    source_signatures: list[str] = []
    snapshot_inputs: list[CollectionSnapshotInput] = []
    for row in manifest_rows:
        source = row.document_artifact
        if (
            row.artifact_id != artifact.pk
            or row.collection_id != int(artifact.scope_id)
            or source.scope_type != GraphArtifact.ScopeType.DOCUMENT
            or source.status != GraphArtifact.Status.ACTIVE
            or source.scope_id != str(row.document_id)
        ):
            raise CollectionResolutionPersistenceError(
                "collection manifest ownership or active source changed"
            )
        expected_source = collection_input_source_signature(
            collection_id=row.collection_id,
            document_id=row.document_id,
            document_artifact=source,
        )
        expected_build = collection_input_build_signature(
            source_signature=expected_source,
            destination_artifact=artifact,
        )
        if (
            row.source_signature != expected_source
            or row.build_signature != expected_build
        ):
            raise CollectionResolutionPersistenceError(
                "collection manifest signature drift detected"
            )
        source_signatures.append(expected_source)
        snapshot_inputs.append(
            CollectionSnapshotInput(
                manifest_input_id=row.pk,
                document_artifact_id=source.pk,
                document_id=row.document_id,
                source_signature=expected_source,
                build_signature=expected_build,
            )
        )
    expected_source_hash = collection_manifest_source_hash(source_signatures)
    if artifact.source_hash != expected_source_hash:
        raise CollectionResolutionPersistenceError(
            "collection artifact source hash does not match exact manifest"
        )
    metadata = artifact.metadata if type(artifact.metadata) is dict else {}
    ontology_checksum = metadata.get("ontology_checksum")
    try:
        _require_hash(ontology_checksum, "artifact ontology checksum")
    except ValueError as exc:
        raise CollectionResolutionPersistenceError(str(exc)) from exc
    return CollectionBuildSnapshot(
        destination_artifact_id=artifact.pk,
        collection_id=int(artifact.scope_id),
        inputs=tuple(snapshot_inputs),
        source_hash=artifact.source_hash,
        ontology_checksum=ontology_checksum,
    )


def _load_resolution_source_rows(artifact, manifest_rows, *, for_update: bool = False):
    from apps.knowledge_graph.models import (
        DocumentEntity,
        DocumentEntityMention,
        RelationMention,
    )

    manifest_by_artifact = {row.document_artifact_id: row for row in manifest_rows}
    source_entity_query = DocumentEntity.objects.filter(
        artifact_id__in=manifest_by_artifact,
        status=DocumentEntity.Status.ACTIVE,
    ).order_by("pk")
    if for_update:
        source_entity_query = source_entity_query.select_for_update()
    source_entities = tuple(source_entity_query)
    membership_query = (
        DocumentEntityMention.objects.select_related("mention")
        .filter(
            document_entity__in=source_entities,
            status=DocumentEntityMention.Status.ACTIVE,
        )
        .order_by("document_entity_id", "mention_id")
    )
    if for_update:
        membership_query = membership_query.select_for_update()
    memberships = tuple(membership_query)
    memberships_by_entity: dict[int, list[object]] = defaultdict(list)
    mention_owner: dict[int, int] = {}
    for membership in memberships:
        memberships_by_entity[membership.document_entity_id].append(membership)
        mention_owner[membership.mention_id] = membership.document_entity_id
    projected: list[DocumentEntityInput] = []
    for entity in source_entities:
        manifest = manifest_by_artifact.get(entity.artifact_id)
        if manifest is None or entity.document_id != manifest.document_id:
            raise CollectionResolutionPersistenceError(
                "document entity escaped the exact collection manifest"
            )
        entity_memberships = memberships_by_entity.get(entity.pk, [])
        if not entity_memberships:
            raise CollectionResolutionPersistenceError(
                "active document entity has no active mention membership"
            )
        aliases = tuple(
            AliasEvidence(
                alias=membership.mention.raw_text,
                method=membership.method,
                mention_id=membership.mention_id,
            )
            for membership in entity_memberships
            if membership.method in _ALIAS_METHODS
            and normalize_entity_label(membership.mention.raw_text).key
            != normalize_entity_label(entity.normalized_label).key
        )
        extraction_confidence = sum(
            membership.mention.extraction_confidence
            for membership in entity_memberships
        ) / len(entity_memberships)
        metadata = entity.metadata if type(entity.metadata) is dict else {}
        description = metadata.get("description", "")
        if type(description) is not str:
            description = ""
        projected.append(
            DocumentEntityInput(
                entity_id=entity.pk,
                document_cluster_key=entity.cluster_key,
                document_artifact_id=entity.artifact_id,
                document_id=entity.document_id,
                label=entity.label,
                normalized_label=entity.normalized_label,
                entity_type=entity.entity_type,
                identifier=entity.identifier,
                version_signature=entity.version_signature,
                alias_evidence=aliases,
                description=description,
                extraction_confidence=extraction_confidence,
                document_resolution_confidence=entity.resolution_confidence,
            )
        )
    relation_query = RelationMention.objects.filter(
        artifact_id__in=manifest_by_artifact,
        head_id__in=mention_owner,
        tail_id__in=mention_owner,
    ).order_by("pk")
    if for_update:
        relation_query = relation_query.select_for_update()
    relation_rows = tuple(relation_query)
    supported_relations = tuple(
        SupportedRelation(
            relation_id=row.pk,
            source_entity_id=mention_owner[row.head_id],
            relation_type=row.relation_type,
            target_entity_id=mention_owner[row.tail_id],
            confidence=row.extraction_confidence,
            supported=True,
        )
        for row in relation_rows
        if mention_owner[row.head_id] != mention_owner[row.tail_id]
    )
    return tuple(projected), supported_relations


def load_collection_resolution_inputs(artifact_id: int, build_run_id: int):
    """Load only exact manifest entities/relations after locking build identity."""

    from django.db import transaction

    from apps.knowledge_graph.models import (
        CollectionArtifactInput,
        GraphArtifact,
        GraphBuildRun,
    )

    _positive_int(artifact_id, "artifact id")
    _positive_int(build_run_id, "build run id")
    with transaction.atomic():
        artifact = GraphArtifact.objects.select_for_update().get(pk=artifact_id)
        run = GraphBuildRun.objects.select_for_update().get(pk=build_run_id)
        _validate_collection_destination(artifact, run)
        manifest = tuple(
            CollectionArtifactInput.objects.select_for_update()
            .select_related("document_artifact", "collection")
            .filter(artifact=artifact)
            .order_by("document_artifact_id")
        )
        snapshot = _snapshot_from_locked_manifest(artifact, manifest)
        entities, relations = _load_resolution_source_rows(
            artifact, manifest, for_update=True
        )
        return snapshot, entities, relations


def _validate_result_against_source(
    result: CollectionResolutionResult,
    snapshot: CollectionBuildSnapshot,
    source_entities: Sequence[DocumentEntityInput],
    source_relations: Sequence[SupportedRelation],
    artifact,
) -> None:
    if type(result) is not CollectionResolutionResult:
        raise CollectionResolutionPersistenceError(
            "result must be exact CollectionResolutionResult"
        )
    if result.snapshot != snapshot:
        raise CollectionResolutionPersistenceError(
            "result snapshot does not match locked manifest"
        )
    if result.resolver_version != artifact.resolver_version:
        raise CollectionResolutionPersistenceError(
            "result resolver version does not match artifact"
        )
    if result.audit.embedding_model_signature != artifact.embedding_model_signature:
        raise CollectionResolutionPersistenceError(
            "result embedding signature does not match artifact"
        )
    source_ids = tuple(entity.entity_id for entity in source_entities)
    if result.source_entity_ids != source_ids:
        raise CollectionResolutionPersistenceError(
            "result source entity IDs do not match locked source"
        )
    if result.source_entity_fingerprint != _source_entity_fingerprint(source_entities):
        raise CollectionResolutionPersistenceError(
            "result source entity fingerprint drift detected"
        )
    if result.source_relation_fingerprint != _source_relation_fingerprint(
        source_relations
    ):
        raise CollectionResolutionPersistenceError(
            "result source relation fingerprint drift detected"
        )
    partition = tuple(
        entity_id
        for cluster in result.clusters
        for entity_id in cluster.document_entity_ids
    )
    if len(partition) != len(set(partition)) or tuple(sorted(partition)) != source_ids:
        raise CollectionResolutionPersistenceError(
            "result clusters must partition every source entity exactly once"
        )
    if resolution_result_checksum(result) != result.checksum:
        raise CollectionResolutionPersistenceError("result checksum is invalid")
    if (
        result.audit.max_observed_candidate_fanout
        > result.audit.max_candidates_per_entity
    ):
        raise CollectionResolutionPersistenceError("candidate fanout exceeds audit cap")
    _expected_persisted_link_count(result)


def _expected_persisted_link_count(result: CollectionResolutionResult) -> int:
    """Derive the exact auto/candidate/rejected row count from a resolver result."""

    if type(result) is not CollectionResolutionResult:
        raise CollectionResolutionPersistenceError(
            "result must be exact CollectionResolutionResult"
        )
    cluster_by_source: dict[int, str] = {}
    for cluster in result.clusters:
        for source_id in cluster.document_entity_ids:
            if source_id in cluster_by_source:
                raise CollectionResolutionPersistenceError(
                    "result contains duplicate source cluster membership"
                )
            cluster_by_source[source_id] = cluster.cluster_key
    if set(cluster_by_source) != set(result.source_entity_ids):
        raise CollectionResolutionPersistenceError(
            "result clusters do not partition source entities"
        )
    alternatives: dict[int, set[str]] = defaultdict(set)
    for decision in result.decisions:
        if decision.outcome is ResolutionOutcome.AUTOMATIC:
            continue
        try:
            left_cluster = cluster_by_source[decision.left_entity_id]
            right_cluster = cluster_by_source[decision.right_entity_id]
        except KeyError as exc:
            raise CollectionResolutionPersistenceError(
                "resolution decision endpoint is outside source partition"
            ) from exc
        if left_cluster != right_cluster:
            alternatives[decision.left_entity_id].add(right_cluster)
            alternatives[decision.right_entity_id].add(left_cluster)
    alternative_count = sum(
        min(len(targets), result.audit.max_candidates_per_entity)
        for targets in alternatives.values()
    )
    return len(result.source_entity_ids) + alternative_count


def _decision_row_checksum(payload: object) -> str:
    return _hash(payload)


def _row_metadata_without_audit(metadata: object) -> dict[str, object]:
    if type(metadata) is not dict:
        return {}
    return {
        key: value for key, value in metadata.items() if key != "row_audit_checksum"
    }


def _collection_entity_row_audit(row) -> str:
    embedding = None
    if row.embedding is not None:
        embedding = [float(value) for value in row.embedding]
    return _hash(
        {
            "artifact_id": row.artifact_id,
            "collection_id": row.collection_id,
            "cluster_key": row.cluster_key,
            "label": row.label,
            "normalized_label": row.normalized_label,
            "version_signature": row.version_signature,
            "entity_type": row.entity_type,
            "identifier": row.identifier,
            "status": row.status,
            "extraction_confidence": row.extraction_confidence,
            "resolution_confidence": row.resolution_confidence,
            "retrieval_utility": row.retrieval_utility,
            "promotion_confidence": row.promotion_confidence,
            "filter_reason": row.filter_reason,
            "embedding_model_signature": row.embedding_model_signature,
            "embedding_input_hash": row.embedding_input_hash,
            "embedding": embedding,
            "metadata": _row_metadata_without_audit(row.metadata),
        }
    )


def _collection_link_row_audit(row) -> str:
    return _hash(
        {
            "artifact_id": row.artifact_id,
            "manifest_input_id": row.manifest_input_id,
            "document_entity_id": row.document_entity_id,
            "collection_cluster_key": row.collection_entity.cluster_key,
            "score": row.score,
            "identifier_score": row.identifier_score,
            "alias_score": row.alias_score,
            "embedding_similarity": row.embedding_similarity,
            "neighborhood_agreement": row.neighborhood_agreement,
            "method": row.method,
            "resolver_version": row.resolver_version,
            "outcome": row.outcome,
            "candidate_rank": row.candidate_rank,
            "decision_checksum": row.decision_checksum,
            "status": row.status,
            "reason": row.reason,
            "metadata": _row_metadata_without_audit(row.metadata),
        }
    )


def _collection_resolution_marker_is_valid(
    marker: object,
    *,
    artifact,
    result: CollectionResolutionResult,
    entity_count: int,
    automatic_assignment_count: int,
    link_count: int,
) -> bool:
    expected_keys = {
        "version",
        "source_hash",
        "source_entity_fingerprint",
        "source_relation_fingerprint",
        "result_checksum",
        "embedding_model_signature",
        "collection_entity_count",
        "automatic_assignment_count",
        "link_count",
    }
    return bool(
        type(marker) is dict
        and set(marker) == expected_keys
        and type(marker.get("version")) is int
        and marker.get("version") == 1
        and marker.get("source_hash") == artifact.source_hash
        and marker.get("source_entity_fingerprint") == result.source_entity_fingerprint
        and marker.get("source_relation_fingerprint")
        == result.source_relation_fingerprint
        and marker.get("result_checksum") == result.checksum
        and marker.get("embedding_model_signature")
        == artifact.embedding_model_signature
        and type(marker.get("collection_entity_count")) is int
        and marker.get("collection_entity_count") == entity_count
        and type(marker.get("automatic_assignment_count")) is int
        and marker.get("automatic_assignment_count") == automatic_assignment_count
        and type(marker.get("link_count")) is int
        and marker.get("link_count") == link_count
    )


def _existing_collection_resolution(artifact, run, result):
    from apps.knowledge_graph.models import (
        CollectionEntity,
        CollectionEntityDocumentLink,
    )

    stats = run.stats if type(run.stats) is dict else {}
    marker = stats.get("collection_resolution_commit")
    entities = tuple(
        CollectionEntity.objects.filter(artifact=artifact).order_by("cluster_key")
    )
    links = tuple(
        CollectionEntityDocumentLink.objects.select_related("collection_entity")
        .filter(artifact=artifact)
        .order_by("pk")
    )
    expected_link_count = _expected_persisted_link_count(result)
    if marker is None:
        if entities or links:
            raise CollectionResolutionPersistenceError(
                "destination has collection rows without a commit marker"
            )
        return None
    if not _collection_resolution_marker_is_valid(
        marker,
        artifact=artifact,
        result=result,
        entity_count=len(result.clusters),
        automatic_assignment_count=len(result.source_entity_ids),
        link_count=expected_link_count,
    ):
        raise CollectionResolutionPersistenceError(
            "existing collection resolution marker is invalid"
        )
    if len(links) != expected_link_count:
        raise CollectionResolutionPersistenceError(
            "existing collection resolution links do not match result"
        )
    if len(entities) != len(result.clusters):
        raise CollectionResolutionPersistenceError(
            "existing collection entity rows do not match marker"
        )
    if any(
        type(row.metadata) is not dict
        or row.metadata.get("result_checksum") != result.checksum
        or row.metadata.get("row_audit_checksum") != _collection_entity_row_audit(row)
        for row in entities
    ) or any(
        type(row.metadata) is not dict
        or row.metadata.get("result_checksum") != result.checksum
        or row.metadata.get("row_audit_checksum") != _collection_link_row_audit(row)
        for row in links
    ):
        raise CollectionResolutionPersistenceError(
            "existing collection resolution row audit is invalid"
        )
    auto_ids = sorted(
        row.document_entity_id for row in links if row.outcome == row.Outcome.AUTOMATIC
    )
    if auto_ids != list(result.source_entity_ids):
        raise CollectionResolutionPersistenceError(
            "existing automatic assignments do not partition source entities"
        )
    return entities


def _write_collection_resolution(artifact, manifest, source_entities, result):
    from apps.knowledge_graph.models import (
        CollectionEntity,
        CollectionEntityDocumentLink,
    )

    source_by_id = {item.pk: item for item in source_entities}
    manifest_by_document_artifact = {
        item.document_artifact_id: item for item in manifest
    }
    entity_rows = [
        CollectionEntity(
            artifact=artifact,
            collection_id=int(artifact.scope_id),
            cluster_key=cluster.cluster_key,
            label=cluster.label,
            normalized_label=cluster.normalized_label,
            version_signature=cluster.version_signature,
            entity_type=cluster.entity_type,
            identifier=cluster.identifier,
            status=CollectionEntity.Status.ACTIVE,
            extraction_confidence=cluster.extraction_confidence,
            resolution_confidence=cluster.resolution_confidence,
            retrieval_utility=0.0,
            promotion_confidence=0.0,
            filter_reason="",
            embedding_model_signature=cluster.embedding_model_signature or "",
            embedding_input_hash=cluster.embedding_input_hash or "",
            embedding=cluster.embedding,
            metadata={
                "aliases": list(cluster.aliases),
                "document_entity_ids": list(cluster.document_entity_ids),
                "document_cluster_keys": list(cluster.document_cluster_keys),
                "resolution_methods": list(cluster.resolution_methods),
                "result_checksum": result.checksum,
            },
        )
        for cluster in result.clusters
    ]
    for row in entity_rows:
        row.metadata = {
            **row.metadata,
            "row_audit_checksum": _collection_entity_row_audit(row),
        }
    CollectionEntity.objects.bulk_create(entity_rows)
    row_by_source: dict[int, CollectionEntity] = {}
    for cluster, row in zip(result.clusters, entity_rows, strict=True):
        for source_id in cluster.document_entity_ids:
            row_by_source[source_id] = row

    decisions_by_source: dict[int, list[CollectionPairDecision]] = defaultdict(list)
    for decision in result.decisions:
        decisions_by_source[decision.left_entity_id].append(decision)
        decisions_by_source[decision.right_entity_id].append(decision)
    links = []
    for source_id in result.source_entity_ids:
        source = source_by_id[source_id]
        accepted = [
            item
            for item in decisions_by_source.get(source_id, ())
            if item.outcome is ResolutionOutcome.AUTOMATIC
            and row_by_source[item.left_entity_id].pk
            == row_by_source[item.right_entity_id].pk
        ]
        best = max(
            accepted,
            key=lambda item: (
                item.resolution_confidence,
                item.tier.value,
                -item.left_entity_id,
                -item.right_entity_id,
            ),
            default=None,
        )
        payload = {
            "kind": "automatic_assignment",
            "source_id": source_id,
            "cluster_key": row_by_source[source_id].cluster_key,
            "result_checksum": result.checksum,
        }
        links.append(
            CollectionEntityDocumentLink(
                artifact=artifact,
                manifest_input=manifest_by_document_artifact[source.artifact_id],
                document_entity=source,
                collection_entity=row_by_source[source_id],
                score=(1.0 if best is None else best.resolution_confidence),
                identifier_score=(None if best is None else best.identifier_score),
                alias_score=(None if best is None else best.alias_score),
                embedding_similarity=(
                    None if best is None else best.embedding_similarity
                ),
                neighborhood_agreement=(
                    None if best is None else best.neighborhood_agreement
                ),
                method="singleton" if best is None else best.tier.value,
                resolver_version=result.resolver_version,
                outcome=CollectionEntityDocumentLink.Outcome.AUTOMATIC,
                candidate_rank=None,
                decision_checksum=_decision_row_checksum(payload),
                status=CollectionEntityDocumentLink.Status.ACTIVE,
                reason=(
                    "singleton_assignment"
                    if best is None
                    else ";".join(best.reason_codes)
                ),
                metadata={"result_checksum": result.checksum},
            )
        )
    alternatives: dict[tuple[int, str], tuple[CollectionPairDecision, int]] = {}
    for decision in result.decisions:
        if decision.outcome is ResolutionOutcome.AUTOMATIC:
            continue
        for source_id, target_id in (
            (decision.left_entity_id, decision.right_entity_id),
            (decision.right_entity_id, decision.left_entity_id),
        ):
            target = row_by_source[target_id]
            if target.pk == row_by_source[source_id].pk:
                continue
            key = (source_id, target.cluster_key)
            current = alternatives.get(key)
            candidate = (decision, target_id)
            if current is not None:
                current_decision = current[0]
                current_rank = (
                    current_decision.outcome is ResolutionOutcome.CANDIDATE,
                    current_decision.resolution_confidence,
                    current_decision.tier.value,
                    -current_decision.left_entity_id,
                    -current_decision.right_entity_id,
                )
                candidate_rank = (
                    decision.outcome is ResolutionOutcome.CANDIDATE,
                    decision.resolution_confidence,
                    decision.tier.value,
                    -decision.left_entity_id,
                    -decision.right_entity_id,
                )
                if candidate_rank <= current_rank:
                    continue
            alternatives[key] = candidate

    alternatives_by_source: dict[int, list[tuple[CollectionPairDecision, int]]] = (
        defaultdict(list)
    )
    for (source_id, _cluster_key), alternative in alternatives.items():
        alternatives_by_source[source_id].append(alternative)
    for source_id in sorted(alternatives_by_source):
        bounded = sorted(
            alternatives_by_source[source_id],
            key=lambda item: (
                item[0].outcome is not ResolutionOutcome.CANDIDATE,
                -item[0].resolution_confidence,
                item[0].tier.value,
                row_by_source[item[1]].cluster_key,
            ),
        )[: result.audit.max_candidates_per_entity]
        for decision, target_id in bounded:
            source = source_by_id[source_id]
            target = row_by_source[target_id]
            outcome = CollectionEntityDocumentLink.Outcome(decision.outcome.value)
            status = {
                CollectionEntityDocumentLink.Outcome.CANDIDATE: (
                    CollectionEntityDocumentLink.Status.SUPPRESSED
                ),
                CollectionEntityDocumentLink.Outcome.REJECTED: (
                    CollectionEntityDocumentLink.Status.REJECTED
                ),
            }[outcome]
            payload = {
                "kind": outcome,
                "source_id": source_id,
                "target_cluster": target.cluster_key,
                "decision": [decision.left_entity_id, decision.right_entity_id],
                "result_checksum": result.checksum,
            }
            link = CollectionEntityDocumentLink(
                artifact=artifact,
                manifest_input=manifest_by_document_artifact[source.artifact_id],
                document_entity=source,
                collection_entity=target,
                score=decision.resolution_confidence,
                identifier_score=decision.identifier_score,
                alias_score=decision.alias_score,
                embedding_similarity=decision.embedding_similarity,
                neighborhood_agreement=decision.neighborhood_agreement,
                method=decision.tier.value,
                resolver_version=result.resolver_version,
                outcome=outcome,
                candidate_rank=decision.candidate_rank,
                decision_checksum=_decision_row_checksum(payload),
                status=status,
                reason=";".join(decision.reason_codes),
                metadata={"result_checksum": result.checksum},
            )
            link.metadata = {
                **link.metadata,
                "row_audit_checksum": _collection_link_row_audit(link),
            }
            links.append(link)
    for link in links:
        if "row_audit_checksum" not in link.metadata:
            link.metadata = {
                **link.metadata,
                "row_audit_checksum": _collection_link_row_audit(link),
            }
    CollectionEntityDocumentLink.objects.bulk_create(links)
    return tuple(entity_rows), tuple(links)


def persist_collection_resolution(
    artifact_id: int,
    build_run_id: int,
    result: CollectionResolutionResult,
):
    """Atomically write a complete shadow result; never activate or replace state."""

    from django.db import transaction

    from apps.knowledge_graph.models import (
        CollectionArtifactInput,
        DocumentEntity,
        GraphArtifact,
        GraphBuildRun,
    )

    _positive_int(artifact_id, "artifact id")
    _positive_int(build_run_id, "build run id")
    with transaction.atomic():
        artifact = GraphArtifact.objects.select_for_update().get(pk=artifact_id)
        run = GraphBuildRun.objects.select_for_update().get(pk=build_run_id)
        _validate_collection_destination(artifact, run)
        manifest = tuple(
            CollectionArtifactInput.objects.select_for_update()
            .select_related("document_artifact", "collection")
            .filter(artifact=artifact)
            .order_by("document_artifact_id")
        )
        snapshot = _snapshot_from_locked_manifest(artifact, manifest)
        source_entities = tuple(
            DocumentEntity.objects.select_for_update()
            .filter(
                artifact_id__in=snapshot.document_artifact_ids,
                status=DocumentEntity.Status.ACTIVE,
            )
            .order_by("pk")
        )
        projected, source_relations = _load_resolution_source_rows(
            artifact, manifest, for_update=True
        )
        _validate_result_against_source(
            result,
            snapshot,
            projected,
            source_relations,
            artifact,
        )
        existing = _existing_collection_resolution(artifact, run, result)
        if existing is not None:
            return existing
        entity_rows, links = _write_collection_resolution(
            artifact, manifest, source_entities, result
        )
        expected_link_count = _expected_persisted_link_count(result)
        if len(links) != expected_link_count:
            raise CollectionResolutionPersistenceError(
                "persisted link count does not match resolution result"
            )
        marker = {
            "version": 1,
            "source_hash": artifact.source_hash,
            "source_entity_fingerprint": result.source_entity_fingerprint,
            "source_relation_fingerprint": result.source_relation_fingerprint,
            "result_checksum": result.checksum,
            "embedding_model_signature": artifact.embedding_model_signature,
            "collection_entity_count": len(entity_rows),
            "automatic_assignment_count": len(result.source_entity_ids),
            "link_count": expected_link_count,
        }
        stats = run.stats if type(run.stats) is dict else {}
        run.stats = {**stats, "collection_resolution_commit": marker}
        run.save(update_fields=["stats"])
        return entity_rows


__all__ = [
    "COLLECTION_RESOLVER_VERSION",
    "AliasEvidence",
    "CollectionBuildSnapshot",
    "CollectionEmbeddingSession",
    "CollectionEntityCluster",
    "CollectionPairDecision",
    "CollectionResolutionConfig",
    "CollectionResolutionPersistenceError",
    "CollectionResolutionResult",
    "CollectionSnapshotInput",
    "DocumentEntityInput",
    "EmbeddedText",
    "ResolutionAudit",
    "SignedEmbeddingBatch",
    "SupportedRelation",
    "default_collection_embedding_session",
    "build_collection_snapshot",
    "embedding_text_hash",
    "load_collection_resolution_inputs",
    "persist_collection_resolution",
    "resolve_collection_entities",
    "resolution_result_checksum",
]
